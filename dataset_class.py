class MaestroDataset(Dataset):
  # Class for storing MAESTRO dataset, for training and evaluation.
  def __init__(self, data_dir):
      self.data_dir = data_dir
      self.midi_files = self._get_midi_files()
      self.data = [midi_to_pianoroll(f) for f in self.midi_files]

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      # Returns pianoroll of the data at the given index.
      return torch.tensor(self.data[idx])

  def _get_midi_files(self):
        # Collects all midi file paths from the given data directory.
        midi_files = []
        for dirpath, _, filenames in os.walk(self.data_dir):
            for filename in filenames:

                file_path = os.path.join(dirpath, filename)
                midi_files.append(file_path)
        return midi_files

# This path should be where the midi data is stored if downloaded using the 'dataset_downloader' file.
midi_paths='./maestro/maestro-v3.0.0'
# Loads the data into a PyTorch Dataset object for training.
dataset = MaestroDataset(midi_paths)

# Random data value to manually inspect shape
dataset[345].shape # Should be of size [128, 96]
