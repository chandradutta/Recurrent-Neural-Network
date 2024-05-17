import torch
from torch import nn
import socket
import wandb
from wandb.keras import WandbCallback
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
socket.setdefaulttimeout(30)
import copy
import random


wandb.login()
wandb.init(project ='vanillaRNN')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
testCSV = "/kaggle/input/telugu/tel/tel_test.csv"
validCSV = "/kaggle/input/telugu/tel/tel_valid.csv"
trainCSV = "/kaggle/input/telugu/tel/tel_train.csv"

validationData = pd.read_csv(validCSV,header = None)
trainingData = pd.read_csv(trainCSV, header=None)
testData = pd.read_csv(testCSV,header= None)


trainingInput = trainingData[0].to_numpy()
trainingOutput = trainingData[1].to_numpy()
#the size of input and output is 4096
validationInput = validationData[0].to_numpy()
validationOutput = validationData[1].to_numpy()


maxLength = 0
maxLengthElement =''
# Loop through elements in validationInput
for element in validationInput:
    if(True):
        maxLength = max(maxLength,len(element))
    if(maxLength == len(element) and True):
        maxLengthElement=element

maxLength =0 
maxLengthElement =''
# Loop through elements in validationOutput
for element in validationOutput:
    maxLength = max(maxLength,len(element))
    if(maxLength == len(element)):
        maxLengthElement=element    

def preProcessTheData(input, output, validation):
    data = {
        "allCharacters1" : [],
        "inputCharToNum": torch.zeros(len(input),30, dtype=torch.int, device=device),
        "charNumMap1" : {},
        "outputData" : output,
        "numCharMap1" : {},
        "inputData" : input,
        "outputCharToNum": torch.zeros(len(output),23, dtype=torch.int, device=device),
        "inputLength" : 0,
        "allCharacters2" : [],
        "charNumMap2" : {},
        "numCharMap2" : {},
        
        "outputLength" : 0
    }
    
    k = 0 

    m1 = data["charNumMap1"]
    m2 = data["charNumMap2"]
    if validation:
        m1 = preProcessedData["charNumMap1"]
        m2 = preProcessedData["charNumMap2"]

    for i in range(0,len(input)):
        charToNum = []
        input[i] = "{" + input[i] + "}"*(29-len(input[i]))
        for char in (input[i]):
            index = 0
            if(char not in data["allCharacters1"]):
                data["allCharacters1"].append(char)
                if validation:
                    index = m1[char]
                else:
                    index = data["allCharacters1"].index(char)
                data["numCharMap1"][index] = char
                data["charNumMap1"][char] = index
            else:
                if validation:
                    index = m1[char]
                else:
                    index = data["allCharacters1"].index(char)
            
            charToNum.append(index)
            
        charToNum1 = []
        my_tensor = torch.tensor(charToNum,device = device)
        data["inputCharToNum"][k] = my_tensor
        
        output[i] = "{" + output[i] + "}"*(22-len(output[i]))
        for char in (output[i]):
            index = 0
            if(char not in data["allCharacters2"]):
                data["allCharacters2"].append(char)
                if validation:
                    index = m2[char]
                else:
                    index = data["allCharacters2"].index(char)
                data["charNumMap2"][char] = index
                data["numCharMap2"][index] = char
            else:
                if validation:
                    index = m2[char]
                else:
                    index = data["allCharacters2"].index(char)
                

            charToNum1.append(index)
            
        my_tensor1 = torch.tensor(charToNum1,device = device)
        my_tensor2 = my_tensor1
        data["outputCharToNum"][k] = my_tensor1
        
        k+=1
    
    data["outputLength"] = len(data["allCharacters2"])
    totalOutputLen = data["outputLength"]
    data["inputLength"] = len(data["allCharacters1"])
   
        
    return data

preProcessedData = preProcessTheData(copy.copy(trainingInput),copy.copy(trainingOutput), False)


preprocessedValidationData = preProcessTheData(copy.copy(validationInput),copy.copy(validationOutput), True)

class createDataset(Dataset):
    def __getitem__(self, idx):
        outputData = self.target[idx]
        inputData = self.source[idx]
        return inputData, outputData
    def __init__(self, x,y):
        self.source = x
        self.target = y

    def __len__(self):
        return len(self.source)
    
    
    

data = preProcessTheData(copy.copy(trainingInput),copy.copy(trainingOutput), False)

def getDataLoader(mode, batchSize):
    if(mode == 'train'and mode == 'train'):
        dataset = createDataset(data["inputCharToNum"],data['outputCharToNum'])
        return DataLoader(dataset, batch_size=batchSize, shuffle=True)
    else:
        if(mode != 'train'):
            dataset = createDataset(preprocessedValidationData["inputCharToNum"],preprocessedValidationData['outputCharToNum'])
            return  DataLoader(dataset, batch_size=batchSize, shuffle=True)
    


def evaluateModelPerformance(encoder, decoder, batchSize, temporalFusionParameter):
    
    accumulatedValidationAccuracy = 0
    totalValidationLoss = 0
    decoder.eval()
    encoder.eval()
    lossCriterion = nn.NLLLoss()
    dataLoader = getDataLoader("validation", batchSize) # Fetch data loader based on the operation mode

    
    for batchIndex, (sourceSequence, targetSequence) in enumerate(dataLoader):
        
        initialEncoderState = encoder.getInitialState() 
        
        encoderOutput, currentStateAfterEncoding = encoder(sourceSequence, initialEncoderState)

        cumulativeLoss = 0 
        predictedOutputs = []
        sequenceLength = targetSequence.shape[1]

        currentDecoderState = currentStateAfterEncoding
        randomSelectionFactor = random.random()
        for index in range(sequenceLength):

            if(index == 0):
                inputTensorForDecoder = targetSequence[:, index].view(batchSize, 1) # Reshape to match expected dimensions
            else:
                if randomSelectionFactor < temporalFusionParameter:
                    inputTensorForDecoder = targetSequence[:, index].view(batchSize, 1) # Pass current batch element
                elif randomSelectionFactor >= temporalFusionParameter:
                    inputTensorForDecoder = inputTensorForDecoder.view(batchSize, 1) # Pass previously selected element

            decodedOutput, updatedDecoderState = decoder(inputTensorForDecoder, currentDecoderState)
            outputDec = decodedOutput
            topValues, topIndices = decodedOutput.topk(1)  # Retrieve top values and their indices
            inputTensorForDecoder = topIndices.squeeze().detach() # Convert to 1D tensor
            inputTen = inputTensorForDecoder
            predictedOutputs.append(inputTensorForDecoder) # Append softmax values
                    
            decodedOutput = decodedOutput[:, -1, :] # Reduce size from (batchSize*1*embeddingSize) to (batchSize*embeddingSize)

            targetCharacters = targetSequence[:, index] #(batchSize)
            tmepChar = targetCharacters
            targetCharacters = targetCharacters.type(dtype=torch.long)

            cumulativeLoss += lossCriterion(decodedOutput, targetCharacters) # Pass softmax values to target characters

        stackedPredictions = torch.stack(predictedOutputs)
        totalPredictions = 51200
        predictionsMatrix = stackedPredictions.transpose(0, 1) # Transpose to match expected format

        accumulatedValidationAccuracy += (predictionsMatrix == targetSequence).all(dim=1).sum().item() # Sum up matching values
        maxCorrect = 100
        totalValidationLoss += (cumulativeLoss.item()/sequenceLength)

        if(batchIndex % 20 == 0):
            print(f"Batch: {batchIndex}, Loss: {cumulativeLoss.item()/sequenceLength}")
    
    encoder.train()
    decoder.train()
#     print(f"Validation Accuracy: {accumulatedValidationAccuracy/40.96}")
#     print(f"Validation Loss: {totalValidationLoss}")
    print("Logging to wandb")
    wandb.log({'validation_accuracy':validation_accuracy/40.96})
    print("Logging to wandb")
    wandb.log({'validation_loss':validation_loss})   



class Encoder(nn.Module):
    
    def __init__(self, inputDimension, embeddingSize, layerCount, neuronCount, cellType, batchSize):
        super(Encoder, self).__init__()
        self.batchSize = batchSize
        self.embeddingLayer = nn.Embedding(inputDimension, embeddingSize)
        self.size = batchSize

        self.neuronCountInHiddenLayer = neuronCount
        self.min_num_layers = 1
        self.layerStack = layerCount
        
        
        if(cellType=='LSTM'):
            self.recurrentLayer = nn.LSTM(embeddingSize, neuronCount, num_layers=layerCount, batch_first=True)
            self.max_num_layers = 4
        elif(cellType == 'RNN'):
            self.recurrentLayer = nn.RNN(embeddingSize, neuronCount, num_layers=layerCount, batch_first=True)
        elif(cellType=='GRU'):
            self.recurrentLayer = nn.GRU(embeddingSize, neuronCount, num_layers=layerCount, batch_first=True)
       
    
    def getInitialState(self):
        tens = torch.zeros(self.layerStack, self.batchSize, self.neuronCountInHiddenLayer, device=device)
        return tens
            
    def forward(self, inputData, previousState):
        embeddedInput = self.embeddingLayer(inputData)
        output, stateAfterPassing = self.recurrentLayer(embeddedInput, previousState)
        return output, stateAfterPassing
    
    
class Decoder(nn.Module):
    def __init__(self, outputDimension, embeddingSize, neuronCount, layerCount, cellType, dropoutProbability):
        super(Decoder, self).__init__()
        self.embeddingLayerForOutput = nn.Embedding(outputDimension, embeddingSize)
        
        if(cellType=="LSTM"):
            self.recurrentLayerForOutput = nn.LSTM(embeddingSize, neuronCount, num_layers=layerCount, batch_first=True)
            self.max_num_layers = 4
        elif(cellType == 'RNN'):
            self.recurrentLayerForOutput = nn.RNN(embeddingSize, neuronCount, num_layers=layerCount, batch_first=True)
        elif(cellType=="GRU"):
            self.recurrentLayerForOutput = nn.GRU(embeddingSize, neuronCount, num_layers=layerCount, batch_first=True)
        
            
        self.applySoftMax = nn.LogSoftmax(dim=2)
        self.finalLinearTransformation = nn.Linear(neuronCount, outputDimension)
        self.dropOutLayer = nn.Dropout(dropoutProbability)

    def forward(self, currentInput, previousState):
        embeddedCurrentInput = self.embeddingLayerForOutput(currentInput)
        tempInput = embeddedCurrentInput
        processedInput = F.relu(embeddedCurrentInput)
        outputFromRecurrent, stateAfterProcessing = self.recurrentLayerForOutput(processedInput, previousState)
        outputFromRecurrent = self.dropOutLayer(outputFromRecurrent)
        tempOutput = outputFromRecurrent
        finalOutput = self.applySoftMax(self.finalLinearTransformation(outputFromRecurrent))
        return finalOutput, stateAfterProcessing

def num_to_char_converter(source_array, target_array, data):
    target_string = ''
    num_to_char_map = data['numCharMap2']
    source_string = ''
    
    for source_row, target_row in zip(source_array, target_array):
        source_string = ''
        temp_source = ''
        target_string = ''
        for source_element, target_element in zip(source_row, target_row):
            target_string += num_to_char_map[target_element.item()]
            source_string += num_to_char_map[source_element.item()]
        
        print("Printing string for testing")
        print(source_string, " ", target_string)

def modelTrainingProcess(embeddingSize, encoderLayerCount, decoderLayerCount, hiddenNeuronsPerLayer, cellType, bidirectionalOption, dropoutRate, numberOfTrainingEpochs, batchSizeForTraining, learningRateValue, chosenOptimizer, temporalFusionThreshold):
   
    dataLoaderInstance = getDataLoader("train", batchSizeForTraining) # Data loader setup for training
    
    decoderInstance = Decoder(data["outputLength"], embeddingSize, hiddenNeuronsPerLayer, encoderLayerCount, cellType, dropoutRate).to(device)
    deocoderCpy = decoderInstance
    encoderInstance = Encoder(data["inputLength"], embeddingSize, encoderLayerCount, hiddenNeuronsPerLayer, cellType, batchSizeForTraining).to(device)
    lossCalculationMethod = nn.NLLLoss()
    
    if(chosenOptimizer == 'Adam'):
        optimizerForEncoder = optim.Adam(encoderInstance.parameters(), lr=learningRateValue)
        optimizerForDecoder = optim.Adam(decoderInstance.parameters(), lr=learningRateValue)
    elif(chosenOptimizer == 'Nadam'):
        optimizerForEncoder = optim.NAdam(encoderInstance.parameters(), lr=learningRateValue)
        optimizerForDecoder = optim.NAdam(decoderInstance.parameters(), lr=learningRateValue)
    
    

    for epochIteration in range(numberOfTrainingEpochs):
        
        totalTrainLoss = 0 
        accumulatedTrainAccuracy = 0 
        

        for batchIndex, (sourceDataBatch, targetDataBatch) in enumerate(dataLoaderInstance):
                        
            initialEncoderState = encoderInstance.getInitialState() 
            
            if(bidirectionalOption == "Yes"):
                flippedSourceBatch = torch.flip(sourceDataBatch, dims=[1]) # Reverse the batch along rows
                sourceDataBatch = (sourceDataBatch + flippedSourceBatch)//2 # Average reversed and original batches
                
            encodedOutput, currentStateAfterEncoding = encoderInstance(sourceDataBatch, initialEncoderState)
            currentBatch = batchIndex+1
            lossAccumulator = 0 
            final_loss = 0
            sequenceLength = targetDataBatch.shape[1]

            predictedSequence = []
            temp_sequence = []
            randomChoiceFactor = random.random()

            currentDecoderState = currentStateAfterEncoding

            for indexPosition in range(sequenceLength):
                
                if(indexPosition == 0):
                    inputTensorForDecoding = targetDataBatch[:, indexPosition].view(batchSizeForTraining, 1) # Reshape to match expected dimensions
                else:
                    if randomChoiceFactor < temporalFusionThreshold:
                        inputTensorForDecoding = targetDataBatch[:, indexPosition].view(batchSizeForTraining, 1) # Pass current batch element
                    elif randomChoiceFactor >= temporalFusionThreshold:
                        inputTensorForDecoding = inputTensorForDecoding.view(batchSizeForTraining, 1) # Pass previously selected element

                decodedResult, updatedDecoderState = decoderInstance(inputTensorForDecoding, currentDecoderState)
                tempResult = decodedResult
                topValuedIndices, topIndexPositions = decodedResult.topk(1)  # Get top values and their indices
               
                inputTensorForDecoding = topIndexPositions.squeeze().detach() # Convert to 1D tensor
                tempInputTensor = inputTensorForDecoding
                predictedSequence.append(inputTensorForDecoding) # Append softmax values
                    
                decodedResult = decodedResult[:, -1, :] # Reduce size from (batchSize*1*embeddingSize) to (batchSize*embeddingSize)

                targetCharacterIndices = targetDataBatch[:, indexPosition] #(batchSize)
                tempTargetchar= targetCharacterIndices
                targetCharacterIndices = targetCharacterIndices.type(dtype=torch.long)

                lossAccumulator += lossCalculationMethod(decodedResult, targetCharacterIndices) # Pass softmax values to target characters

            stackedPredictions = torch.stack(predictedSequence)
            maxPrediction = stackedPredictions
            predictionsMatrix = stackedPredictions.transpose(0, 1) # Transpose to match expected format

            if(batchIndex == 0 and epochIteration == numberOfTrainingEpochs-1):
                num_to_char_converter(targetDataBatch, predictionsMatrix, data) 

            accumulatedTrainAccuracy += (predictionsMatrix == targetDataBatch).all(dim=1).sum().item() # Sum up matching values
            totalTrainLoss += (lossAccumulator.item()/sequenceLength)
            
    
            optimizerForEncoder.zero_grad()
            optimizerForDecoder.zero_grad()
            lossAccumulator.backward()
            optimizerForEncoder.step()
            optimizerForDecoder.step()
            
        # print(f"Train Accuracy: {accumulatedTrainAccuracy/512}")
        # print(f"Train Loss: {totalTrainLoss}")
        wandb.log({'train_accuracy':train_accuracy/512})
        wandb.log({'train_loss':train_loss})
        evaluateModelPerformance(encoderInstance,decoderInstance,batchSizeForTraining,temporalFusionThreshold)



def main_fun():
    wandb.init(project ='vanillaRNN')
    params = wandb.config
    with wandb.init(project = 'vanillaRNN', name='embedding'+str(params.embSize)+'cellType'+params.cellType+'batchSize'+str(params.batchsize)) as run:
        modelTrainingProcess(params.embSize,params.encoderLayers,params.decoderLayers,params.hiddenLayerNuerons,params.cellType,params.bidirection,params.dropout,params.epochs,params.batchsize,params.learningRate,params.optimizer,params.tf_ratio)
    
sweep_params = {
     'name'   : 'DeepLearningAssignment3',
    'method' : 'bayes',
    'metric' : {
         'name' : 'validation_accuracy',
        'goal' : 'maximize',
       
    },
    'parameters' : {
        'tf_ratio' :{'values' : [0.2,0.4,0.5]},
         'dropout' : {'values' : [0,0.2,0.3]},
         'encoderLayers':{'values':[1,5,10]},
        'embSize':{'values':[16,32,64]},
        'hiddenLayerNuerons'   : {'values' : [64,256,512]},
        'cellType' : {'values' : ['GRU'] } ,
        'bidirection' : {'values' : ['no']},
        'batchsize' : {'values' : [32,64]},
        'epochs'  : {'values': [10,20,30]},
        'optimizer':{'values' : ['Adam','Nadam']},
        'learningRate' : {'values' : [1e-2,1e-3,1e-4]},
        'decoderLayers' : {'values' : [1,5,10]},
    }
}
sweepId = wandb.sweep(sweep_params,project = 'vanillaRNN')
wandb.agent(sweepId,function =main_fun,count = 2)
