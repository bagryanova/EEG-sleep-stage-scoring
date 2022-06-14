import mne
import torch
from tqdm.auto import tqdm

N_SUBJECTS = 82
SUBJECTS_BLACK_LIST = [13, 36, 52, 39, 68, 69, 78, 79]

'''
TRAIN_SUBJECTS = [
    55, 31, 34, 72, 50, 42, 60, 25,  4, 56, 66,  7, 54, 58, 20,  9, 15,
]

TEST_SUBJECTS = [19, 27, 59, 30, 48, 40]
'''

TRAIN_SUBJECTS = [
    55, 31, 34, 72, 50, 42, 60, 25,  4, 56, 66,  7, 54, 58, 20,  9, 15,
    46, 75, 14, 65, 33, 18, 26, 64, 21, 38, 81, 28,  3, 73, 51,  6,
    10, 11, 61, 37, 63, 41, 17, 70,  1, 77, 62, 76, 44, 57,  2,  5, 43,
    23, 74, 53, 45
]

TEST_SUBJECTS = [19, 27, 59, 30, 48, 40, 22, 24, 32, 67, 47, 35, 16, 12, 8, 80, 29, 71, 49, 0]

def read_epochs(file, annotation_file, chunk_duration=30.0):
    data = mne.io.read_raw_edf(file, stim_channel='Event marker', verbose='WARNING')
    annotations = mne.read_annotations(annotation_file)
    data.set_annotations(annotations, emit_warning=False, verbose='WARNING')

    class_map = dict({
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
    })

    events, _ = mne.events_from_annotations(data, event_id=class_map, chunk_duration=chunk_duration, verbose='WARNING')

    event_id = dict({
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3/4': 3,
        'Sleep stage R': 4
    })

    tmax = chunk_duration - 1. / data.info['sfreq']

    picks = mne.pick_types(data.info, eeg=True)
    epochs = mne.Epochs(
        raw=data,
        events=events,
        picks=picks,
        preload=True,
        event_id=event_id,
        tmin=0.,
        tmax=tmax,
        baseline=None,
        on_missing='warn',
        verbose='WARNING')

    return epochs.get_data(), epochs.events[:, 2]

def fetch_edf_files(subjects, recordings):
    for s in subjects:
        if s in SUBJECTS_BLACK_LIST:
            raise ValueError("Requested subject from black list")

    return mne.datasets.sleep_physionet.age.fetch_data(subjects=subjects, recording=recordings)

def download_and_preprocess(subjects):
    files = fetch_edf_files(subjects, [1, 2])

    X = []
    y = []
    sequence = []

    for i, f in enumerate(tqdm(files)):
        s, l = read_epochs(*f)

        cur_X = torch.tensor(s[:, 0, :]).float()
        cur_y = torch.tensor(l)

        w_edge_mins = 30
        nw_idx = torch.where(cur_y != 0)[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(cur_y): end_idx = len(cur_y) - 1
        select_idx = torch.arange(start_idx, end_idx+1)

        # print('Before crop:', cur_X.shape, cur_y.shape)

        cur_X = cur_X[select_idx]
        cur_y = cur_y[select_idx]

        # print('After crop:', cur_X.shape, cur_y.shape)

        X.append(cur_X)
        y.append(cur_y)
        sequence.append(i * torch.ones(cur_X.shape[0], dtype=torch.int64))

    X = torch.cat(X, axis=0)
    y = torch.cat(y)
    sequence = torch.cat(sequence)

    return X, y, sequence
