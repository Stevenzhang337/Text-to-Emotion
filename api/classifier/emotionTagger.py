from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class EmotionTagger():
  def __init__(self, model, tokenizer, device):
    self.tokenizer = tokenizer
    self.device = device
    self.model = model
  
  def _tokenize(self, dataset):
    #token texts
    initTokenizedData = []
    labels = []
    for i in range(len(dataset)):
      text = dataset[i]
      initTokenizedData.append("[CLS] " + text + " [SEP]")
      labels.append(-1)
    tokenizedTexts = [self.tokenizer.tokenize(trainText) for trainText in initTokenizedData]

    return tokenizedTexts, labels
    
  def _getInputs(self, tokenizedData):
    #encode text as integers
    maxLen = 128
    inputIds = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenizedData]
    inputIds = pad_sequences(inputIds, maxlen=maxLen, dtype="long", truncating="post", padding="post")

    #create attention masks
    attentionMasks = []
    for seq in inputIds:
      seqMask = [float(i>0) for i in seq]
      attentionMasks.append(seqMask)
  
    return inputIds, attentionMasks

  def _getDataLoader(self, inputIds, labels, attentionMask, batchSize):
    #encode into tensor model
    tensorInput = torch.tensor(inputIds)
    tensorLabels =  torch.tensor(labels)
    tensorMasks = torch.tensor(attentionMask)

    #create iterators with torch dataloader (for saving memory)
    tensorData = TensorDataset(tensorInput, tensorMasks, tensorLabels)
    #tensorSampler = RandomSampler(tensorData)
    dataLoader = DataLoader(tensorData, batch_size=batchSize)

    return dataLoader

  #predict on unknown text
  def evaluate(self, text):
    results = []
    dataset = [text]
    tokenizedData, _ = self._tokenize(dataset)
    inputIds, attentionMask = self._getInputs(tokenizedData)

    dataLoader = self._getDataLoader(inputIds, _, attentionMask, 32)

    self.model.eval()


    for batch in dataLoader:
      batch = tuple(t.to(self.device) for t in batch)
      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, _ = batch #np.array(tensorInput[i]), np.array(tensorMasks[i])

      with torch.no_grad():
        # Forward pass, calculate logit predictions
        result = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      
      logits = result.logits
      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      results.append(logits)
  
    maxResult = []
    for i in range(len(results)):
      for logits in results[i]:
        maxResult.append(np.argmax(logits))

    return maxResult[0], logits


output_dir = 'classifier/model_save/'

# Load a trained model and vocabulary that you have fine-tuned
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)

# Copy the model to the GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

emotionModel = EmotionTagger(model, tokenizer, device)

def getModel():
  return emotionModel
