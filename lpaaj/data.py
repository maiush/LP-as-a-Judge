import ast
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict

from datasets import load_dataset
from lpaaj.constants import DATA_PATH


class TextDataset():

    def __init__(
            self,
            task: str,
            dataset: str,
            prompt_key: str,
            label_key: Optional[str] = None,
            reverse: Optional[bool] = False,
            contrast_choice: Optional[str] = None
    ) -> None:
        if task == "score":
            assert dataset in ["newsroom", "summeval", "hanna"]

        self.dataset = dataset
        self.prompt_key = prompt_key
        self.label_key = label_key
        self.task = task
        self.reverse = reverse
        self.contrast_choice = contrast_choice

        self.items = {
            'newsroom': 'summary',
            'summeval': 'summary',
            'hanna': 'story',
            'rocstories': 'answer'
        }

        self.aspects_noun2adj = {
            'informativeness': 'informative',
            'relevance': 'relevant',
            'fluency': 'fluent',
            'coherence': 'coherent',
            'consistency': 'consistent',
            'empathy': 'empathetic',
            'surprise': 'surprising',
            'engagement': 'engaging',
            'complexity': 'complex'
        }

        # read data
        path = f"{DATA_PATH}/{dataset}/{dataset}_prompts"
        if task == "score":
            path = f"{path}_zero_shot.jsonl"
        elif task in ["compare", "contrast"]:
            path = f"{path}_compare"
            if reverse:
                path = f"{path}_reversed"
            path = f"{path}.jsonl"
        print(f"loading data from {path}")
        self.data = pd.read_json(path, orient="records", lines=True)
        self.prompts = self.data[prompt_key].tolist()
        # read labels
        path = f"{DATA_PATH}/{dataset}/{dataset}"
        if task == "score" or dataset == "rocstories":
            path = f"{path}.jsonl"
        elif task in ["compare", "contrast"]:
            path = f"{path}_pairwise_comparisons.jsonl"
        print(f"loading labels from {path}")
        self.labels = pd.read_json(path, orient="records", lines=True)[label_key if label_key else prompt_key].tolist()  

    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.prompts[idx], self.labels[idx]

    def get_prompt(self, prompt: str) -> List[Dict[str, str]]:
        user_prompt = {
            'role': 'user',
            'content': prompt
        }
        assistant_prompt = {
            'role': 'assistant',
            'content': self.get_assistant_prompt()
        }
        prompt = [user_prompt, assistant_prompt]
        return prompt

    def get_assistant_prompt(self) -> str:
        if self.dataset not in ["caters", "mctaco"]: 
            item = self.items[self.dataset]
        if self.task == "score":
            content = f"I would rate the {self.prompt_key} as a "
            return content
        elif self.dataset == "caters":
            content = 'Between statement 1 and statement 2, the statement which appears before the other is statement'
        elif self.dataset == "mctaco":
            content = "Between choice 1 and choice 2, the more sensible option is choice"
        else:
            aspect = self.aspects_noun2adj[self.prompt_key]
            content = f"Between {item} 1 and {item} 2, the more {aspect} choice is {item}"
        if self.task == "contrast":
            content += self.contrast_choice
        return content

    def preprocess_prompts(
        self,
        tokenizer
    ) -> None:
        self.tokenizer = tokenizer
        apply_chat_template = self.tokenizer.apply_chat_template

        self.processed_prompts = []
        for prompt in tqdm(self.prompts, desc="preprocessing data"):
            messages = self.get_prompt(prompt)
            prompt = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = self.check_priming(messages, prompt)
            self.processed_prompts.append(prompt)
        self.prompts = self.processed_prompts

    def check_priming(
            self,
            messages: List[Dict[str, str]],
            prompt: str
    ) -> str:
        '''
        if we want to prime the model, we need to deal with special tokens
        '''
        # this only applies if we're priming the assistant
        if messages[-1]['role'] != 'assistant': return prompt
        message = messages[-1]['content']
        # we need to handle the case where the last character is a space
        space = message[-1] == ' '
        if space: message = message[:-1]
        # we need to chop off the chat template tags added by the tokenizer
        ix = prompt.rindex(message) + len(message)
        prompt = prompt[:ix]
        # add the space back in necessary
        if space: prompt = prompt + ' '
        return prompt


class MTBench:
    def __init__(
            self,
            task: str,
            reverse: Optional[bool] = False,
            contrast_choice: Optional[str] = None
    ) -> None:
        self.task = task
        self.reverse = reverse
        self.contrast_choice = contrast_choice
        # Load dataset
        data = load_dataset("lmsys/mt_bench_human_judgments", split="human", trust_remote_code=True).to_pandas()
        # Convert winner to 1, -1, 0
        data["winner"] = data["winner"].map({"model_a": 1, "model_b": -1, "tie": 0})
        # Drop rows where judge starts with "author"
        data = data[~data["judge"].str.startswith("author")]
        # Drop judge column
        data = data.drop(columns=["judge"])
        # Convert conversation_a and conversation_b to strings
        data["conversation_a"] = data["conversation_a"].apply(lambda x: str(x))
        data["conversation_b"] = data["conversation_b"].apply(lambda x: str(x))
        # For conversation_a and conversation_b, add a comma after every close curly brace
        data['conversation_a'] = data['conversation_a'].apply(lambda x: x.replace('}', '},'))
        data['conversation_b'] = data['conversation_b'].apply(lambda x: x.replace('}', '},'))
        # Group by all columns except winner and sum the winner column
        data = data.groupby(['question_id', 'model_a', 'model_b', 'conversation_a', 'conversation_b', 'turn']).winner.sum().reset_index()
        # Convert winner values based on sign and drop zeros
        data = data[data['winner'] != 0]  # Drop rows where winner is 0
        data['winner'] = (data['winner'] < 0).astype(int) + 1  # Convert to 1 or 2
        # Convert conversation_a and conversation_b back to lists
        data['conversation_a'] = data['conversation_a'].apply(lambda x: ast.literal_eval(x))
        data['conversation_b'] = data['conversation_b'].apply(lambda x: ast.literal_eval(x))
        # Filter conversations for turn 1 to keep only first two elements
        data.loc[data["turn"] == 1, "conversation_a"] = data.loc[data["turn"] == 1, "conversation_a"].apply(lambda x: x[:2])
        data.loc[data["turn"] == 1, "conversation_b"] = data.loc[data["turn"] == 1, "conversation_b"].apply(lambda x: x[:2])
        # Convert conversations to strings
        def conversation_to_string(conversation, model_letter):
            result = []
            for i, msg in enumerate(conversation):
                if i % 2 == 0:  # User messages are at even indices
                    result.append(f'User: "{msg["content"]}"')
                else:  # Assistant messages are at odd indices
                    result.append(f'Model {model_letter}: "{msg["content"]}"')
            return '\n'.join(result)
        # Apply conversion to both columns
        data['conversation_a'] = data['conversation_a'].apply(lambda x: conversation_to_string(x, '1'))
        data['conversation_b'] = data['conversation_b'].apply(lambda x: conversation_to_string(x, '2'))
        # Reset index
        data = data.reset_index(drop=True)

        self.data = data
        self.prompts = self.get_prompts()
        self.labels = self.get_labels()

    def __getitem__(self, index: int) -> str:
        return self.prompts[index]

    def __len__(self) -> int:
        return len(self.prompts)

    def get_labels(self) -> list[str]:
        return self.data['winner'].tolist()

    def get_prompts(self) -> list[str]:
        template = """\
Here are conversations between a user and two models.

----------
{conversation_a}
----------

----------
{conversation_b}
----------

Evaluate the quality of the responses provided by the two models. Which followed the user's instructions and answers the user's questions better? Answers must be a single choice."""
        out = []
        for i in range(len(self.data)):
            a, b = self.data["conversation_a"][i], self.data["conversation_b"][i]
            if self.reverse: a, b = b, a
            out.append(template.format(conversation_a=a, conversation_b=b).strip())
        return out

    def get_prompt(self, prompt: str) -> List[Dict[str, str]]:
        user_prompt = {
            'role': 'user',
            'content': prompt
        }
        assistant_prompt = {
            'role': 'assistant',
            'content': self.get_assistant_prompt()
        }
        prompt = [user_prompt, assistant_prompt]
        return prompt

    def get_assistant_prompt(self) -> str:
        content = "Between Model 1 and Model 2, the better answer is from Model"
        if self.task == "contrast":
            content += self.contrast_choice
        return content
    
    def preprocess_prompts(
        self,
        tokenizer
    ) -> None:
        self.tokenizer = tokenizer
        apply_chat_template = self.tokenizer.apply_chat_template

        self.processed_prompts = []
        for prompt in tqdm(self.prompts, desc="preprocessing data"):
            messages = self.get_prompt(prompt)
            prompt = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = self.check_priming(messages, prompt)
            self.processed_prompts.append(prompt)
        self.prompts = self.processed_prompts

    def check_priming(
            self,
            messages: List[Dict[str, str]],
            prompt: str
    ) -> str:
        '''
        if we want to prime the model, we need to deal with special tokens
        '''
        # this only applies if we're priming the assistant
        if messages[-1]['role'] != 'assistant': return prompt
        message = messages[-1]['content']
        # we need to handle the case where the last character is a space
        space = message[-1] == ' '
        if space: message = message[:-1]
        # we need to chop off the chat template tags added by the tokenizer
        ix = prompt.rindex(message) + len(message)
        prompt = prompt[:ix]
        # add the space back in necessary
        if space: prompt = prompt + ' '
        return prompt


class LLMBar:
    VALID_LLMBAR_SUBSETS = {
        "Adversarial_GPTInst", "Adversarial_GPTOut", "Adversarial_Neighbor", "Adversarial_Manual", "Natural",
        "Constraint", "Negation", "Normal", "Base_9", "Base_10"
    }
    
    def __init__(self, subset: str, task: str, reverse: Optional[bool] = False, contrast_choice: Optional[str] = None) -> None:
        if subset not in self.VALID_LLMBAR_SUBSETS:
            raise ValueError(
                f"Invalid subset: {subset}. Valid values are: {', '.join(sorted(self.VALID_LLMBAR_SUBSETS))}"
            )
        self.task = task
        self.reverse = reverse
        self.contrast_choice = contrast_choice
        self.subset = subset
        # load the appropriate subset
        try:
            self.data = load_dataset("princeton-nlp/LLMBar", "LLMBar", trust_remote_code=True)
            self.data = self.data[subset].to_pandas()
        except:
            self.data = load_dataset("princeton-nlp/LLMBar", "CaseStudy", trust_remote_code=True)
            self.data = self.data[subset].to_pandas()
        self.prompts = self.get_prompts()
        self.labels = self.get_labels()

    def __getitem__(self, index: int) -> str:
        return self.prompts[index]

    def __len__(self) -> int:
        return len(self.data)

    def get_labels(self) -> list[str]:
        return self.data['label'].tolist()
    
    def get_prompts(self) -> list[str]:
        template = """\
{input}
Choice 1: {output_1}
Choice 2: {output_2}

Which choice is better? Answers must be a single choice."""
        out = []
        for i in range(len(self.data)):
            a, b = self.data["output_1"][i], self.data["output_2"][i]
            if self.reverse: a, b = b, a
            out.append(template.format(input=self.data["input"][i], output_1=a, output_2=b).strip())
        return out

    def get_prompt(self, prompt: str) -> List[Dict[str, str]]:
        user_prompt = {
            'role': 'user',
            'content': prompt
        }
        assistant_prompt = {
            'role': 'assistant',
            'content': self.get_assistant_prompt()
        }
        prompt = [user_prompt, assistant_prompt]
        return prompt

    def get_assistant_prompt(self) -> str:
        content = "Between Choice 1 and Choice 2, the better answer is Choice"
        if self.task == "contrast":
            content += self.contrast_choice
        return content
    
    def preprocess_prompts(
        self,
        tokenizer
    ) -> None:
        self.tokenizer = tokenizer
        apply_chat_template = self.tokenizer.apply_chat_template

        self.processed_prompts = []
        for prompt in tqdm(self.prompts, desc="preprocessing data"):
            messages = self.get_prompt(prompt)
            prompt = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = self.check_priming(messages, prompt)
            self.processed_prompts.append(prompt)
        self.prompts = self.processed_prompts

    def check_priming(
            self,
            messages: List[Dict[str, str]],
            prompt: str
    ) -> str:
        '''
        if we want to prime the model, we need to deal with special tokens
        '''
        # this only applies if we're priming the assistant
        if messages[-1]['role'] != 'assistant': return prompt
        message = messages[-1]['content']
        # we need to handle the case where the last character is a space
        space = message[-1] == ' '
        if space: message = message[:-1]
        # we need to chop off the chat template tags added by the tokenizer
        ix = prompt.rindex(message) + len(message)
        prompt = prompt[:ix]
        # add the space back in necessary
        if space: prompt = prompt + ' '
        return prompt