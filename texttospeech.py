import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer
#importing LJ Speech SEt
#id: name of the corresponding .wav file//transcription:words spoken by the reader//normalised transcription: transcription with nuumbers ordinals and monetary units expanded into full words
data_url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
data_path=keras.utils.get_file(fname="LJSpeech-1.1",origin=data_url,untar=True)
wavs_path=data_path+"\wavs"
metadata_path=data_path+"\metadata.csv"
metadata=pd.read_csv(metadata_path,sep="|",header=None,quoting=3)
metadata.columns=["filename","transcription","normalised_transcription"]
metadata=metadata[["filename","normalised_transcription"]]
metadata=metadata.sample(frac=1).reset_index(drop=True)
#train-test split
split=int(len(metadata)*0.90)
df_train=metadata[:split]
df_test=metadata[split:]
#preprocessing
characters=[x for x in "abcdefghijklmnopqrstuvwxyz'?! "]#list of acceptable characters
char_to_num=keras.layers.StringLookup(vocabulary=characters,oov_token="")#characters to  integers
num_to_char=keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(),oov_token="",invert=True)#int to char
#window length specs
frame_length=256#window length in samples
frame_step=160#number of sample steps
fft_length=384#size of the fourier transform to apply
def encode_sample(wav_file,label):
    file=tf.io.read_file(wavs_path+wav_file+ ".wav")#reading file
    audio,_=tf.audio.decode_wav(file)#decoding file to float tensor
    audio=tf.squeeze(audio,axis=1)#removes repeating elements
    audio=tf.cast(audio,tf.float32)
    spectogram=tf.signal.stft(audio,frame_length=frame_length,frame_step=frame_step,fft_length=fft_length)#getting the spectogram
    spectogram=tf.abs(spectogram)
    spectogram=tf.math.pow(spectogram,0.5)#getting the magnitude of the signal
    #normalisation 
    mean=tf.math.reduce_mean(spectogram,1,keepdims=True)
    stddev=tf.math.reduce_std(spectogram,1,keepdims=True)
    spectogram=(spectogram-mean)/(stddev+1e-10)
    label=tf.strings.lower(str(label))
    label=tf.strings.unicode_split(label, input_encoding="UTF-8")
    label=char_to_num(label)
    return spectogram,label
#creating a dataset to yield the transformed elements as in input
batch_size=32
train_dataset=tf.data.Dataset.from_tensor_slices((list(df_train["filename"]),list(df_train["normalised_transcription"])))
train_dataset=(train_dataset.map(encode_sample,num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
test_dataset=tf.data.Dataset.from_tensor_slices((list(df_test["filename"]),list(df_test["normalised_transcription"])))
test_dataset=(train_dataset.map(encode_sample,num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
#visualisation of an element from the dataset
fig=plt.figure(figsize=(8,5)) 
for batch in train_dataset.take(1):
    spectogram=batch[0][0].numpy()
    spectogram=np.array([np.trim_zeros(x)for x in np.transpose(spectogram)])
    #spectogram
    label=batch[1][0]
    label=tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
    ax=plt.subplot(2,1,1)
    ax.imshow(spectogram,vmax=1)
    ax.set_title(label)
    ax.axis("off")
    #wav file
    file=tf.io.read_file(wavs_path+list(df_train["filename"])[0]+".wav")
    audio,_=tf.audio.decode_wav(file)
    audio=audio.numpy()
    ax=plt.subplot(2,1,2)
    plt.plot(audio)
    ax.set_title("Signal wave")
    ax.set_xlim(0,len(audio))
    display.display(display.Audio(np.transpose(audio),rate=16000))
plt.show()
#building the model
#defining the ctc loss function
def CTCLoss(y_true,y_pred):
    #computing the training time loss value
    batch_len=tf.cast(tf.shape(y_true)[0],dtype="int64")
    input_len=tf.cast(tf.shape(y_pred)[1],dtype="int64")                                                              
    label_len=tf.cast(tf.shape(y_true)[1],dtype="int64")    
    input_len=input_len* tf.ones(shape=(batch_len,1),dtype="int64")                                                
    label_len=label_len* tf.ones(shape=(batch_len,1),dtype="int64")  
    loss=keras.backend.ctc_backend_cost(y_true,y_pred,input_len,label_len)
    return loss
#defining the model
def build_model(input_dim,output_dim,rnn_layers=5,rnn_units=120):
    input_spectogram=layers.Input((None,input_dim),name="input")
    #expand the dimension to use 2d cnn
    x=layers.Reshape((-1,input_dim,1),name="expand_dim")(input_spectogram)
    #convolution layer 1
    x=layers.Conv2D(
        filters=32,
        kernel_size=[11,41],
        strides=[2,2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x) 
    x=layers.BatchNormalization(name="conv_1_bn")(x)
    x=layers.ReLU(name="conv_1_relu")(x)
    #convolution layer 2
    x=layers.Conv2D(
        filters=32,
        kernel_size=[11,21],
        strides=[1,2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x) 
    x=layers.BatchNormalization(name="conv_2_bn")(x)
    x=layers.ReLU(name="conv_2_relu")(x)                      
    #reshape the obtained volume to fit the rnn layers
    x=layers.Reshape((-1,x.shape[-1]*x.shape[-2]))(x)
    #rnn layers
    for i in range(1,rnn_layers+1):
        recurrent=layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        ) 
        x=layers.Bidirectional(recurrent,name=f"Bidirectional_{i}",merge_mode="concat")(x)
        if i<rnn_layers:
            x=layers.Dropout(rate=0.5)(x)
    #dense layer
    x=layers.Dense(units=rnn_units*2,name="dense_1")(x)
    x=layers.ReLU(name="dense_1_relu")(x)
    x=layers.Dropout(rate=0.5)(x)
    #classification layer
    output=layers.Dense(units=output_dim+1,activation="softmax")(x)
    #model
    model=keras.Model(input_spectogram,output,name="text_to_speech")
    #optimization
    opt=keras.optimizers.Adam(learning_rate=1e-4)
    #compiling the model
    model.compile(optimizer=opt,loss=CTCLoss)
    return model                                     
#get the model
model=build_model(
    input_dim=fft_length//2+1,
    output_dim=char_to_num.vocabulary_size(),
    rnn_units=512,
)
model.summary(line_length=110)
#training and evaluating
def decode_batch_prediction(pred):
    input_len=np.ones(pred.shape[0])*pred.shape[1]
    #using greedy search
    results=keras.backend.ctc_decode(pred, input_length=input_len,greedy=True)[0][0]
    output_text=[]
    for result in results:
        result=tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text
#a callback class to output a few trancriptions while training
class CallbackEval(keras.callbacks.Callback):
    def __init__(self,dataset):
        super().__init__()
        self.dataset=dataset
    def on_epoch_end(self,epoch:int,logs=None):
        predictions=[]
        targets=[]
        for batch in self.dataset:
            x,y=batch
            batch_predictions=model.predict(x)
            batch_predictions=decode_batch_prediction(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label=(
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score=wer(targets,predictions)
        print("-"*100)
        print(f"word error rate:{wer_score:.4f}")
        print("-"*100)
        for i in np.random.randint(0,len(predictions),2):
            print(f"Target:  {targets[i]}")
            print(f"Prediction:  {predictions[i]}")
            print("-"*100)
#training process
#defining number of epochs
epochs=2
validation_callback=CallbackEval(test_dataset)
history=model.fit(
    train_dataset,
    test_data=test_dataset,
    epochs=epochs,
    callbacks=[validation_callback],
)
#inference
predictions=[]
targets=[]
for batch in test_dataset:
    x,y=batch
    batch_predictions=model.predict(x)
    batch_predictions=decode_batch_prediction(batch_predictions)
    predictions.extend(batch_predictions)
    for label in y:
        label=(
            tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        )
        targets.append(label)
wer_score=wer(targets,predictions)
print("-"*100)
print(f"word error rate:{wer_score:.4f}")
print("-"*100)
for i in np.random.randint(0,len(predictions),2):
    print(f"Target:  {targets[i]}")
    print(f"Prediction:  {predictions[i]}")
    print("-"*100)

            
        