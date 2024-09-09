"""You should implement both of these methods.

Vanilla edits should just be a custom generate loop with a huggingface
transformer.

Speculative edits should implement the speculative editing algorithm.

To test these, make sure they work on the prompt provided in the README"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def vanilla_edit(prompt: str, max_tokens: int) -> str:
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + max_tokens,
        num_return_sequences=1,
        attention_mask=attention_mask,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def speculative_edit(prompt: str, max_tokens: int) -> str:
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated = input_ids[0].tolist()

    def sample(logits, temperature=0.7):
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).item()

    def speculative_step(prefix, draft_tokens):
        with torch.no_grad():
            prefix_ids = torch.tensor([prefix])
            outputs = model(prefix_ids)
            base_logits = outputs.logits[:, -1, :]

            accepted = 0
            for token in draft_tokens:
                p = torch.softmax(base_logits, dim=-1)[0, token].item()
                if p > 0.01:  # Acceptance threshold
                    accepted += 1
                    next_input = torch.cat([prefix_ids, torch.tensor([[token]])], dim=1)
                    outputs = model(next_input)
                    base_logits = outputs.logits[:, -1, :]
                else:
                    break

            next_token = sample(base_logits)
            return draft_tokens[:accepted] + [next_token], accepted < len(draft_tokens)

    # Extract the code block from the prompt
    code_start = prompt.find("```") + 3
    code_end = prompt.rfind("```")
    original_code = prompt[code_start:code_end].strip()

    # Prepare the edit instruction (in this case, adding a comment)
    edit_instruction = "# Add a single comment\n"
    draft_tokens = tokenizer.encode(edit_instruction, add_special_tokens=False)

    while len(generated) - len(input_ids[0]) < max_tokens:
        prefix = generated
        new_tokens, continue_edit = speculative_step(prefix, draft_tokens)
        generated.extend(new_tokens)

        if not continue_edit:
            break

        # Reset draft tokens after applying the edit
        draft_tokens = []

    result = tokenizer.decode(generated, skip_special_tokens=True)

    # Reconstruct the prompt structure
    edited_code = result[len(prompt) - len(original_code):]
    final_result = prompt[:code_start] + edited_code + prompt[code_end:]

    return final_result

# Test function
def test_edit_functions():
    test_prompt = '''Please add a single comment
```ts
export default function Visualization() {
  const [instanceIdInputs, setInstanceIdInputs] = createSignal<
    InstanceId[] | null
  >(null);
  const [storedInput, setStoredInput] = createSignal<string>("");
  const [datapointOptions, setDatapointOptions] = createSignal<PropsInstance[]>(
    []
  );
  const [shouldRefreshGold, setShouldRefreshGold] =
    createSignal<boolean>(false);
  const [showGold, setShowGold] = createSignal<boolean>(false);
  const [selectedGoldRequestId, setSelectedGoldRequestId] = createSignal<
    string | undefined
  >(undefined);
  const [goldInstances, setGoldInstances] = createSignal<
    {
      sessionId: string;
      email: string | undefined;
      requestId: string | undefined;
      dateAdded: Date;
      type: $Enums.CppGoldExampleType;
    }[]
  >([]);
}
```
```ts
'''

    print("Vanilla Edit Result:")
    print(vanilla_edit(test_prompt, 50))

    print("\nSpeculative Edit Result:")
    print(speculative_edit(test_prompt, 50))

# Run the test
if __name__ == "__main__":
    test_edit_functions()
