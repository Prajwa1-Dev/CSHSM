# M5.1 Source Audit Report

## Overview
This document summarizes the sources acquired and audited for M5.1.

### Multitarget-CONAN
- **Status**: Acquired
- **Local Path**: `data/counter_speech_kb/sources\Multitarget-CONAN.csv`
- **Size**: 1183029 bytes
- **Checksum (SHA-256)**: `1ab624c9e74ac4ef4f4913991802f7305b4a0bad21b8693e4d61ce22c874da26`
- **License**: CC BY-NC-SA 4.0
- **Acquisition Date**: 2026-10-04T15:58:08.775990Z
- **URL**: https://raw.githubusercontent.com/marcoguerini/CONAN/master/Multitarget-CONAN/Multitarget-CONAN.csv
- **Record Count**: 5003
- **Columns**: INDEX, HATE_SPEECH, COUNTER_NARRATIVE, TARGET, VERSION
- **Duplicates/Empty**: 0 / 0

### CONAN
- **Status**: Acquired
- **Local Path**: `data/counter_speech_kb/sources\CONAN.json`
- **Size**: 5791637 bytes
- **Checksum (SHA-256)**: `511c062b5563affbc78bb2c9d9edafd88fe6419add73b5190865bb42863eacc4`
- **License**: CC BY-NC-SA 4.0
- **Acquisition Date**: 2026-10-04T15:58:09.960420Z
- **URL**: https://raw.githubusercontent.com/marcoguerini/CONAN/master/CONAN/CONAN.json
- **Record Count**: 14988
- **Columns**: cn_id, hateSpeech, counterSpeech, hsType, hsSubType, cnType, age, gender, educationLevel
- **Duplicates/Empty**: 0 / 0

### KanHope
- **Status**: Unavailable
- **Reason**: HTTP Error 404: Not Found

### IndicCONAN
- **Status**: Unavailable
- **Reason**: Not publicly available on GitHub/HuggingFace. Requires contacting authors (AAAI 2024). Proceeding without it as per M5.1.

## Storage Footprint
**Total Data Downloaded**: 6.65 MB

## Recommendation
M5.1 is complete. The approved primary sources (Multitarget-CONAN and CONAN) were successfully acquired and audited. IndicCONAN is unavailable but is not a blocker. M5.2 may proceed when authorized.
