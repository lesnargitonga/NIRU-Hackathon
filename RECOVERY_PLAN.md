# 🛑 Operation Clean Slate: Stabilization Plan

You are right. The project has pivoted from AirSim to PX4/Gazebo, but the codebase is cluttered with old files, making it confusing. We are going to fix this **right now**.

## 🚩 Phase 1: Sanitation (The "Cleanup")
**Goal:** Remove noise so you only see what matters.
- [ ] Create a `legacy_airsim/` folder.
- [ ] Move `airsim/` folder and `airsim-env/` into `legacy_airsim/`.
- [ ] Move any `Lesnar AI 1/` duplicate folders to a `backup/` location or delete if safe.
- [ ] Ensure the root only contains: `training`, `backend`, `frontend`, `docker`, and configuration.

## 🚩 Phase 2: Git Hygiene (The "Push")
**Goal:** Secure your work so far.
- [ ] Check `git status` (Done - analyzing results).
- [ ] Create a robust `.gitignore` to stop pushing virtual environments (`.venv`, `backend-env`) and massive datasets.
- [ ] Commit the "Stability Patch" and the new README.
- [ ] Push to the submission repo to secure the checkpoint.

## 🚩 Phase 3: The Data Pipeline (The "Database")
**Goal:** Ensure the "Database" (Dataset) is real and valid.
- [ ] Verify `dataset/` folder structure exists.
- [ ] Confirm `px4_teacher_collect_gz.py` is writing CSVs that we can actually use.
- [ ] **Action:** We will generate a "Gold Standard" dataset sample today to prove the pipeline works.

## 🚩 Phase 4: The Submission Narrative
**Goal:** A clear story for the judges.
- [ ] Align `submission_description.txt` with the new "God Mode" reality.
- [ ] Ensure the "Demo" instructions in README are bulletproof.

---
**Current Status:** I am analyzing your file structure and Git history now. Wait for my report.
