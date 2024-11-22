import tflite_runtime.interpreter as tflite
from tflite_runtime import interpreter as tflite_interpreter
from tflite_runtime.interpreter import load_delegate
import numpy as np
import joblib
from scipy import signal

# Load the scaler
scaler = joblib.load('/home/varunthanneeru/Downloads/scaler.pkl')

# Preprocessing functions (same as before)
def window_data(data, window_size, overlap):
    step_size = int(window_size * (1 - overlap))
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def convert_to_frequency_domain(data, fs):
    fft_data = np.fft.fft(data, axis=0)
    frequencies = np.fft.fftfreq(data.shape[0], d=1/fs)
    positive_freq_indices = frequencies > 0  # Select only positive frequencies
    positive_frequencies = np.abs(fft_data[positive_freq_indices])
    limited_frequencies = positive_frequencies[:31]
    return limited_frequencies

# Load the TFLite model
model_path = "/home/varunthanneeru/Downloads/cnn_lstm_model.tflite"

# Load the Flex delegate (ensure the .so file is available)
flex_delegate = load_delegate('libtensorflowlite_flex_delegate.so')

# Create an interpreter with the Flex delegate
interpreter = tflite_interpreter.Interpreter(model_path=model_path, custom_op_registerers=[flex_delegate])
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Load the dataset
data = np.genfromtxt('/home/varunthanneeru/Downloads/EEG_dataset/up-13.csv', delimiter=',', skip_header=1)[:, 1:]

# Preprocessing pipeline
windowed_data = window_data(data, 127, 0.5)
print("Shape after windowing:", windowed_data.shape)

filtered_data = np.array([apply_bandpass_filter(window, 0.1, 60, 256) for window in windowed_data])
print("Shape after filtering:", filtered_data.shape)

frequency_domain_data = np.array([convert_to_frequency_domain(window, 256) for window in filtered_data])
print("Shape after frequency domain conversion:", frequency_domain_data.shape)

# Normalize data using the scaler
for i in range(frequency_domain_data.shape[2]):  # Iterate over channels
    reshaped_data = frequency_domain_data[:, :, i].reshape(-1, frequency_domain_data.shape[1])  # Reshape for scaling
    scaled_data = scaler.transform(reshaped_data)  # Apply scaler
    frequency_domain_data[:, :, i] = scaled_data.reshape(frequency_domain_data[:, :, i].shape)  # Reshape back

# Prepare input for the model
padded_data = np.zeros((frequency_domain_data.shape[0], 127, frequency_domain_data.shape[2]))
padded_data[:, :31, :] = frequency_domain_data  # Copy the original data into the padded array

# Reshape data for the model
input_data = padded_data.reshape(padded_data.shape[0], 127, 4, 1).astype(np.float32)

# Set input tensor to the model
interpreter.set_tensor(input_details[0]['index'], input_data)

# Perform inference
interpreter.invoke()

# Get predictions
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_classes = np.argmax(output_data, axis=1)

# Map predictions to labels
class_mapping = {0: 'down', 1: 'up', 2: 'left', 3: 'right', 4: 'forward', 5: 'back'}
predicted_labels = [class_mapping[i] for i in predicted_classes]

print("Predicted labels:", predicted_labels)

# Determine the most frequent prediction
majority_vote = list(class_mapping.keys())[np.argmax(np.bincount(predicted_classes))]
print("Final predicted class:", class_mapping[majority_vote])