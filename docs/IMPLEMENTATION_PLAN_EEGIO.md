## Stage 1: Data format confirmation
**Goal**: Confirm EEG-IO CSV and labels structure for one subject.
**Success Criteria**: Sample data file loads (time, Fp1, Fp2), labels file parses corrupt intervals and blink events.
**Tests**: Manually load one subject using `read_data.py` logic.
**Status**: Complete

## Stage 2: Raw loader + event parsing
**Goal**: Implement `_find_files` and `_read_raw` for EEG-IO CSVs only.
**Success Criteria**: MNE RawArray created with correct channel names, sampling rate (250 Hz), and units (uV → V).
**Tests**: Build raw for subject 1 without errors.
**Status**: Complete

## Stage 3: Trial/segment extraction + labels
**Goal**: Convert blink events into trials/segments with 1s windows centered on blink time.
**Success Criteria**: Segments labeled [0,1,2,3] and “without blink” windows drawn from non-blink intervals.
**Tests**: Validate segment counts and label distribution for subject 1.
**Status**: Complete

## Stage 4: Corrupt interval handling + amplitude scaling
**Goal**: Exclude or mark segments overlapping corrupt intervals and adapt amplitude threshold (~300000 uV).
**Success Criteria**: Corrupt intervals are skipped and amplitude threshold does not reject most data.
**Tests**: Build subject 1 and confirm rejection rate is reasonable.
**Status**: Complete

## Stage 5: Metadata + end-to-end sanity check
**Goal**: Align dataset metadata (20 subjects, 250 Hz, Fp1/Fp2) and build one HDF5.
**Success Criteria**: `sub_1.h5` created and dataset_info.json saved.
**Tests**: Run builder for subject 1.
**Status**: Complete
