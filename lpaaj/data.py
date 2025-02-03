import pandas as pd
from lpaaj.constants import DATA_DIR
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict
    

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
        path = f"{DATA_DIR}/{dataset}/{dataset}_prompts"
        if task == "score":
            path = f"{path}_zero_shot.jsonl"
        elif task in ["compare", "contrast"]:
            path = f"{path}_compare"
            if reverse:
                path = f"{path}_reversed"
            path = f"{path}.jsonl"
        self.data = pd.read_json(path, orient="records", lines=True)
        self.prompts = self.data[prompt_key].tolist()
        # read labels
        path = f"{DATA_DIR}/{dataset}/{dataset}"
        if task == "score":
            path = f"{path}.jsonl"
        elif task in ["compare", "contrast"]:
            path = f"{path}_pairwise_comparisons.jsonl"
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
            content = 'Between statement 1 and statement 2, the statement which appears before the other is statement '
        elif self.dataset == "mctaco":
            content = "Between choice 1 and choice 2, the more sensible option is choice "
        else:
            aspect = self.aspects_noun2adj[self.prompt_key]
            content = f"Between {item} 1 and {item} 2, the more {aspect} choice is {item} "
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