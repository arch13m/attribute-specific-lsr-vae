# File originally created in Colab. The next two lines download libraries that aren't pre-installed in Colab.
#!pip install pretty_midi
#!pip install torchinfo

# Library dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio
import pretty_midi
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from torchinfo import summary
import matplotlib.pyplot as plt
from google.colab import files
import IPython
from IPython.core.display import display
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from google.colab import files
import random

# Constants
BATCH_SIZE = 98
LATENT_DIM = 64

# Helper functions for modal pitch 

def get_modal_pitch(pianoroll):
  # Returns modal pitch for a given sample
  mode = [0, 0]
  # For each pitch, count how many notes are played
  for i in range(pianoroll.shape[0]):
    count = 0
    for j in range(pianoroll.shape[1]):
      if pianoroll[i, j] == 1:
        if j == 0 or pianoroll[i, j-1] == 0:
          count += 1
    # Keep track of the pitch with the most notes played
    if count > mode[1]:
      mode = [i, count]

  # most_common returns [pitch, count], we only need the pitch
  return mode[0]
  # NB: if there are multiple pitches that are played the same amount of times, it will only return the lowest of them.

def pitch_num_to_name(mode):
  # Returns the most common pitch in a readable format
  octave = mode // 12 - 2 # MIDI octaves begin at -2
  note_num = mode % 12
  note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  return note_names[note_num] + str(int(octave))

# Helper function for rhythmic complexity

def get_rhythmic_complexity(pianoroll):
  # Returns rhythmic complexity for a given sample
  count = 0
  for i in range(pianoroll.shape[1]):
    for j in range(pianoroll.shape[0]):
      if (pianoroll[j, i] != 0) & (pianoroll[j, i-1] == 0):
        count += 1
        break

  rhythmic_complexity = count / pianoroll.shape[1]
  return rhythmic_complexity

# Helper functions for switching between data types (MIDI, pianoroll, pretty midi)

def midi_to_pianoroll(file_path, fs=8, max_len=96):
  # Converts midi file to pianoroll format
  pm = pretty_midi.PrettyMIDI(file_path)
  pr = pm.get_piano_roll(fs=fs)
  pr = (pr > 0).astype(np.float32)  # Binary roll (i.e. no velocity information)
  pr = pr[:128, :max_len]  # Truncate
  padded = np.pad(pr, ((0, 0), (0, max(0, max_len - pr.shape[1]))), mode='constant')
  #padded = padded.unsqueeze(0)
  return padded

def piano_roll_to_pretty_midi(piano_roll, fs, program=0):
  # Converts pianoroll to MIDI format for manual inspection
  '''(ported from pretty-midi GitHub examples): https://github.com/craffel/pretty-midi/blob/main/examples/reverse_pianoroll.py
  Convert a Piano Roll array into a PrettyMidi object
  with a single instrument.

  Parameters
  ----------
  piano_roll : np.ndarray, shape=(128,frames), dtype=int
      Piano roll of one instrument
  fs : int
      Sampling frequency of the columns, i.e. each column is spaced apart
      by ``1./fs`` seconds.
  program : int
      The program number of the instrument.

  Returns
  -------
  midi_object : pretty_midi.PrettyMIDI
      A pretty_midi.PrettyMIDI class instance describing
      the piano roll.

  '''
  notes, frames = piano_roll.shape
  pm = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(program=program)

  # pad 1 column of zeros so we can acknowledge inital and ending events
  piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

  # use changes in velocities to find note on / note off events
  velocity_changes = np.nonzero(np.diff(piano_roll).T)

  # keep track on velocities and note on times
  prev_velocities = np.zeros(notes, dtype=int)
  note_on_time = np.zeros(notes)

  for time, note in zip(*velocity_changes):
      # use time + 1 because of padding above
      velocity = piano_roll[note, time + 1]
      time = time / fs
      if velocity > 0:
          if prev_velocities[note] == 0:
              note_on_time[note] = time
              prev_velocities[note] = velocity
      else:
          pm_note = pretty_midi.Note(
              velocity=prev_velocities[note],
              pitch=note,
              start=note_on_time[note],
              end=time)
          instrument.notes.append(pm_note)
          prev_velocities[note] = 0
  pm.instruments.append(instrument)
  return pm

# Helper functions for evaluation: varying latent dimension

def linear_vary_constrained_latent_dim(idx, alpha, attribute):
  # Returns the value of the attribute when [idx]th latent dimension is at [alpha]
  # To be used for linear models
  with torch.no_grad():
    # Add a batch dimension to the input data
    input_data = dataset[idx].unsqueeze(0)
    z_mu, z_logvar = model.encode(input_data)
    z = model.reparameterise(z_mu, z_logvar)
    # Apply the alpha value to the 0th dimension of the latent code
    z[0, 0] += alpha
    recon_z = model.decode(z)
    z_pianoroll = recon_z.view(128, 96).cpu().numpy()
    # Round small numbers down
    z_pianoroll = np.round(z_pianoroll, 2)
    # Round larger numbers up to 1 to create a note
    z_pianoroll = np.ceil(z_pianoroll)
    midi = piano_roll_to_pretty_midi(z_pianoroll, fs=16)
    print(f"Alpha = {alpha}")
    print(f"Most common pitch: {pitch_num_to_name(get_modal_pitch(z_pianoroll))}")
    display(IPython.display.Audio(midi.synthesize(fs=16000), rate=16000))

    if attribute == "mod":
      return(get_modal_pitch(z_pianoroll))
    elif attribute == "rhy":
      return(get_rhythmic_complexity(z_pianoroll))
    else:
      raise ValueError("'attribute' value should be 'mod' or 'rhy'.")

def cnn_vary_constrained_latent_dim(idx, alpha, attribute):
  # Returns the value of the attribute when [idx]th latent dimension is at [alpha]
  # To be used for CNN models
  with torch.no_grad():
    # Add a batch dimension to the input data
    input_data = dataset[idx].unsqueeze(0).unsqueeze(0)
    z_mu, z_logvar = model.encode(input_data)
    z = model.reparameterise(z_mu, z_logvar)
    # Apply the alpha value to the 0th dimension of the latent code
    z[0, 0] += alpha
    recon_z = model.decode(z)
    z_pianoroll = recon_z.view(128, 96).cpu().numpy()
    # Round small numbers down
    z_pianoroll = np.round(z_pianoroll, 2)
    # Round larger numbers up to 1 to create a note
    z_pianoroll = np.ceil(z_pianoroll)
    midi = piano_roll_to_pretty_midi(z_pianoroll, fs=16)
    print(f"Alpha = {alpha}")
    print(f"Most common pitch: {pitch_num_to_name(get_modal_pitch(z_pianoroll))}")
    display(IPython.display.Audio(midi.synthesize(fs=16000), rate=16000))
    if attribute == "mod":
      return(get_modal_pitch(z_pianoroll))
    elif attribute == "rhy":
      return(get_rhythmic_complexity(z_pianoroll))
    else:
      raise ValueError("'attribute' value should be 'mod' or 'rhy'.")

def linear_interpretability_metric(attribute):
  # Returns the latent dimension with the best fit for the given attribute, and the dimension's regression score
  # To be used for linear models
  interpretability_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
  z_all = []
  attr_all = []
  for batch_id, batch in enumerate(interpretability_dataloader):
    mu, logvar = model.encode(batch)
    z_tilde = model.reparameterise(mu, logvar)

    # Iterate through the batch and get the most common pitch for each sample
    batch_attrs = []

    if attribute == "mod":
      for i in range(batch.size(0)):
          batch_attrs.append(get_modal_pitch(batch[i]))
    elif attribute == "rhy":
       for i in range(batch.size(0)):
          batch_attrs.append(get_rhythmic_complexity(batch[i]))
    else:
      raise ValueError("'attribute' value should be 'mod' or 'rhy'.")

    attr = torch.tensor(batch_attrs, device=batch.device, dtype=torch.float32)

    z_all.append(z_tilde.cpu().detach().numpy())
    attr_all.append(attr.cpu().detach().numpy())

  z_all = np.concatenate(z_all)
  attr_all = np.concatenate(attr_all)

  mutual_info = np.zeros(LATENT_DIM)
  for i in range(LATENT_DIM):
      mutual_info[i] = mutual_info_score(z_all[:, i], attr_all)
  dim = np.argmax(mutual_info)
  max_mutual_info = np.max(mutual_info)

  reg = LinearRegression().fit(z_all[:, dim:dim+1], attr_all)
  score = reg.score(z_all[:, dim:dim+1], attr_all)
  print("Highest scoring dimension: ", dim)
  print("Regression score: ", score)

def cnn_interpretability_metric(attribute):
  # Returns the latent dimension with the best fit for the given attribute, and the dimension's regression score
  # To be used for CNN models
  interpretability_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
  z_all = []
  attr_all = []
  for batch_id, batch in enumerate(interpretability_dataloader):
    batch = batch.unsqueeze(1)
    mu, logvar = model.encode(batch)
    z_tilde = model.reparameterise(mu, logvar)

    batch = batch.squeeze(1)

    # Iterate through the batch and get the most common pitch for each sample
    batch_attrs = []
    if attribute == "mod":
      for i in range(batch.size(0)):
          batch_attrs.append(get_modal_pitch(batch[i]))
    elif attribute == "rhy":
       for i in range(batch.size(0)):
          batch_attrs.append(get_rhythmic_complexity(batch[i]))
    else:
      raise ValueError("'attribute' value should be 'mod' or 'rhy'.")

    attr = torch.tensor(batch_attrs, device=batch.device, dtype=torch.float32)

    z_all.append(z_tilde.cpu().detach().numpy())
    attr_all.append(attr.cpu().detach().numpy())

  z_all = np.concatenate(z_all)
  attr_all = np.concatenate(attr_all)

  mutual_info = np.zeros(LATENT_DIM)
  for i in range(LATENT_DIM):
      mutual_info[i] = mutual_info_score(z_all[:, i], attr_all)
  dim = np.argmax(mutual_info)
  max_mutual_info = np.max(mutual_info)

  reg = LinearRegression().fit(z_all[:, dim:dim+1], attr_all)
  score = reg.score(z_all[:, dim:dim+1], attr_all)
  print("Highest scoring dimension: ", dim)
  print("Regression score: ", score)
