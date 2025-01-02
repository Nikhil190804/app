import numpy as np
import opensmile
import pickle
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from scipy.signal import butter, lfilter
from scipy.io import wavfile


class App(App):
    def build(self):
        # Initialize OpenSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.model = self.load_model("5s_Logistic_Regression.pkl")
        return self.root
    
    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def open_file_chooser(self):
        print("Opening file chooser")  # Debugging message
        filechooser = FileChooserIconView()
        popup = Popup(
            title="Select an Audio File",
            content=filechooser,
            size_hint=(0.9, 0.9),
        )
        filechooser.bind(
        on_submit=lambda fc, selection, touch: self.on_file_selected(selection, popup)
            )
        popup.open()

    def on_file_selected(self, selection, popup):
        popup.dismiss()  # Close the popup
        if selection and isinstance(selection, list):  # Ensure selection is a non-empty list
            file_path = selection[0]
            print(f"Selected file: {file_path}")  # Debugging message
            try:
                self.process_audio(file_path,popup)  # Process the selected file
            except Exception as e:
                print(f"Error processing file: {e}")
                self.root.ids.status_label.text = f"Error: {e}"
        else:
            print("No file selected.")  # Debugging message
            self.root.ids.status_label.text = "No file selected."



    def process_audio(self, file_path, popup):
        popup.dismiss()
        status_label = self.root.ids.status_label
        status_label.text = "Processing..."
        try:
            audio = self.load_audio(file_path)
            processed_audio = self.preprocess_audio(audio)
            features = self.extract_features(file_path)
            prediction = self.predict(features)
            status_label.text = f"Prediction: {prediction}"
        except Exception as e:
            status_label.text = f"Error: {e}"

    def load_audio(self, file_path):
        rate, audio = wavfile.read(file_path)
        return audio

    def preprocess_audio(self, audio):
        def bandpass_filter(data, lowcut=100, highcut=3500, fs=16000, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype="band")
            return lfilter(b, a, data)

        filtered_audio = bandpass_filter(audio)
        filtered_audio = self.add_white_noise(filtered_audio)
        return filtered_audio

    def add_white_noise(self, audio, noise_level=0.01):
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise

    def extract_features(self, file_path):
        features = self.smile.process_file(file_path)
        a=features.to_numpy().flatten()
        print(a)
        return features.to_numpy().flatten()

    def predict(self, features):
        return self.model.predict([features])[0]

    
    
    def load_model(self, model_path):
        from joblib import load
        return load(model_path)


if __name__ == "__main__":
    App().run()
