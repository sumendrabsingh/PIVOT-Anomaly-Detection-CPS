# False Negative Dataset Investigation Report

This report investigates the specific sensor values during False Negative (FN) samples, comparing them against the normal training distribution.

## Stage1 Analysis
**Total FN Samples:** 12432

### Out-of-Bounds Sensor Values
These sensors exhibited values during FNs that fall outside the observed normal training range (could indicate saturation or unseen states):
| Sensor | Normal Range (Train) | FN Range (Test) |
|---|---|---|
| LIT101 | [120.6237, 817.5565] | [189.8263, 925.0323] |


---

## Stage2 Analysis
**Total FN Samples:** 12432

### Out-of-Bounds Sensor Values
These sensors exhibited values during FNs that fall outside the observed normal training range (could indicate saturation or unseen states):
| Sensor | Normal Range (Train) | FN Range (Test) |
|---|---|---|
| AIT203 | [312.2789, 567.4699] | [287.9518, 335.6575] |
| AIT201 | [251.6662, 272.5263] | [169.3156, 263.5222] |
| FIT201 | [0.0000, 2.4879] | [0.0000, 2.8269] |


---

## Stage6 Analysis
**Total FN Samples:** 12432

---
