from transformers import AutoTokenizer

# --------------------------------------------------
# Load Marathi BERT tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/storage/divya/exam/embeddings/marathi-bert-v2",
    use_fast=False
)

# --------------------------------------------------
# Files
# --------------------------------------------------

input_file = "/mnt/storage/divya/exam/bbmodel/model_3.valid.pred"

output_file = "/mnt/storage/divya/exam/bbmodel/prediction.detok"

# --------------------------------------------------
# Detokenize
# --------------------------------------------------

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:

        tokens = line.strip().split()

        # convert subword tokens -> normal sentence
        text = tokenizer.convert_tokens_to_string(tokens)

        # remove extra spaces
        text = text.strip()

        fout.write(text + "\n")

print("Detokenized file saved to:")
print(output_file)
