# Requirements Document

## Introduction

This feature expands the antenna-ml prediction pipeline in three major areas: (1) broadening the supported frequency range from 2–7 GHz to 1–10 GHz and adding optional dimension constraints as model inputs, (2) allowing users to select or upload training datasets from within the Gradio UI and trigger retraining, and (3) packaging the entire application as a standalone binary executable that bundles the Gradio UI, trained models, and all dependencies.

## Glossary

- **App**: The antenna-ml Gradio web application that serves predictions and manages datasets.
- **Prediction_Engine**: The subsystem that loads a trained model, scales inputs, and returns antenna parameter predictions.
- **Training_Pipeline**: The subsystem that loads a dataset, trains a Random Forest model, evaluates it, and persists model artifacts.
- **Dataset_Manager**: The subsystem responsible for listing, selecting, uploading, and validating training datasets within the App.
- **Binary_Packager**: The build tooling (e.g., PyInstaller) that bundles the Python application, dependencies, and assets into a standalone executable.
- **Dimension_Constraint**: An optional user-supplied bound (minimum and/or maximum) on one or more antenna design parameters (e.g., patch length, substrate width) used to filter or condition predictions.
- **Dataset_Registry**: A configuration structure that maps a human-readable dataset name to its file path and metadata (frequency range, column schema).

## Requirements

### Requirement 1: Expanded Frequency Range

**User Story:** As an antenna engineer, I want to predict antenna parameters for frequencies from 1 to 10 GHz, so that I can design antennas for a wider range of applications beyond WiFi 7.

#### Acceptance Criteria

1. THE App SHALL accept frequency inputs in the range 1.0 to 10.0 GHz inclusive.
2. WHEN a user enters a frequency between 1.0 and 10.0 GHz, THE Prediction_Engine SHALL return predictions for all antenna design parameters.
3. WHEN a user enters a frequency below 1.0 GHz or above 10.0 GHz, THE Prediction_Engine SHALL display a warning indicating the frequency is outside the supported range.
4. THE App SHALL update the frequency slider minimum to 1.0 GHz and maximum to 10.0 GHz.
5. THE config.py module SHALL define FREQ_MIN as 1.0 and FREQ_MAX as 10.0.

### Requirement 2: Optional Dimension Constraints

**User Story:** As an antenna engineer, I want to optionally specify dimension constraints (e.g., maximum patch length, substrate width bounds), so that predictions are filtered to physically feasible designs for my application.

#### Acceptance Criteria

1. THE App SHALL provide optional input fields for each antenna design parameter allowing the user to specify a minimum value, a maximum value, or both.
2. WHEN no Dimension_Constraints are provided, THE Prediction_Engine SHALL return the unconstrained prediction (current behavior).
3. WHEN one or more Dimension_Constraints are provided, THE Prediction_Engine SHALL check each predicted parameter against the supplied bounds.
4. WHEN a predicted parameter falls outside a user-supplied Dimension_Constraint, THE App SHALL flag that parameter in the results output with a warning indicator.
5. THE App SHALL clearly label each Dimension_Constraint input field with the parameter name and unit (mm or mm²).
6. WHEN a user provides a minimum value greater than the corresponding maximum value for a Dimension_Constraint, THE App SHALL display a validation error and refuse to run the prediction.

### Requirement 3: Dataset Selection from the App

**User Story:** As a developer, I want to select which training dataset to use from within the Gradio UI, so that I can retrain the model on different datasets without editing code or config files.

#### Acceptance Criteria

1. THE App SHALL display a dropdown listing all datasets registered in the Dataset_Registry.
2. THE Dataset_Registry SHALL be stored as a JSON configuration file in the project directory.
3. WHEN the App starts, THE Dataset_Manager SHALL scan the Dataset_Registry and populate the dropdown with available dataset names.
4. WHEN a user selects a dataset from the dropdown and triggers training, THE Training_Pipeline SHALL train a new model using the selected dataset.
5. WHEN training completes, THE App SHALL reload the newly trained model artifacts and use them for subsequent predictions without requiring a restart.
6. THE App SHALL display training progress and results (R² score, MSE) after training completes.

### Requirement 4: Dataset Upload

**User Story:** As a developer, I want to upload new CSV datasets through the Gradio UI, so that I can add future datasets without manual file management.

#### Acceptance Criteria

1. THE App SHALL provide a file upload component that accepts CSV files.
2. WHEN a CSV file is uploaded, THE Dataset_Manager SHALL validate that the file contains the expected column schema (frequency column plus antenna parameter columns).
3. WHEN validation succeeds, THE Dataset_Manager SHALL copy the file to the project datasets directory and register it in the Dataset_Registry.
4. WHEN validation fails, THE Dataset_Manager SHALL display a descriptive error message listing the specific schema violations.
5. IF the uploaded file has the same name as an existing dataset, THEN THE Dataset_Manager SHALL reject the upload and inform the user that a dataset with that name already exists.
6. WHEN a new dataset is successfully uploaded, THE App SHALL refresh the dataset dropdown to include the new entry.

### Requirement 5: Dataset Validation

**User Story:** As a developer, I want uploaded datasets to be validated before they are accepted, so that malformed data does not corrupt the training pipeline.

#### Acceptance Criteria

1. THE Dataset_Manager SHALL verify that the uploaded CSV contains a column named "Frequency(GHz)".
2. THE Dataset_Manager SHALL verify that all columns in the uploaded CSV are numeric.
3. THE Dataset_Manager SHALL verify that the uploaded CSV contains no NaN values.
4. THE Dataset_Manager SHALL verify that the uploaded CSV contains at least 10 rows of data.
5. IF any validation check fails, THEN THE Dataset_Manager SHALL return a ValidationResult containing all detected errors.
6. THE Dataset_Manager SHALL log all validation warnings and errors using the Python logging module.

### Requirement 6: Binary Executable Packaging

**User Story:** As a user, I want to download and run the antenna prediction app as a standalone executable, so that I do not need to install Python, Docker, or any dependencies.

#### Acceptance Criteria

1. THE Binary_Packager SHALL produce a single-file or single-directory executable that bundles the App, all Python dependencies, and the Gradio UI assets.
2. WHEN the executable is launched, THE App SHALL start the Gradio server and open the prediction interface on a local port.
3. THE Binary_Packager SHALL include any pre-trained model artifacts (.pkl files) present at build time inside the executable bundle.
4. THE Binary_Packager SHALL include a default dataset inside the executable bundle so that the App can function without external files.
5. WHEN no pre-trained model is bundled, THE App SHALL prompt the user to select a dataset and train a model before predictions are available.
6. THE executable SHALL run on the build platform's operating system without requiring Python or pip to be installed.
7. THE Binary_Packager SHALL be invocable via a build script or Makefile target (e.g., `./run.sh build-exe`).

### Requirement 7: Model Retraining from the App

**User Story:** As a developer, I want to trigger model retraining from the Gradio UI after selecting a dataset, so that the full workflow (select data, train, predict) is self-contained in the app.

#### Acceptance Criteria

1. THE App SHALL provide a "Train Model" button on the dataset management tab.
2. WHEN the "Train Model" button is clicked, THE Training_Pipeline SHALL train a new model using the currently selected dataset.
3. WHILE training is in progress, THE App SHALL display a status indicator showing that training is running.
4. WHEN training completes successfully, THE App SHALL display the model evaluation metrics (R² score, MSE, MAE per parameter).
5. IF training fails due to a dataset error or runtime exception, THEN THE App SHALL display the error message and retain the previously loaded model.
6. THE Training_Pipeline SHALL save model artifacts with a prefix derived from the dataset name to avoid overwriting models trained on other datasets.

### Requirement 8: Configuration Consistency

**User Story:** As a developer, I want all new configuration values (frequency range, dataset paths, packaging settings) managed through config.py, so that the single-source-of-truth convention is maintained.

#### Acceptance Criteria

1. THE config.py module SHALL define the expanded frequency range constants (FREQ_MIN = 1.0, FREQ_MAX = 10.0).
2. THE config.py module SHALL define a DATASETS_DIR path constant pointing to the directory where uploaded datasets are stored.
3. THE config.py module SHALL define a DATASET_REGISTRY_PATH constant pointing to the dataset registry JSON file.
4. THE config.py module SHALL define all new default values as named constants rather than inline literals.
5. WHEN a new configuration constant is added, THE config.py module SHALL include a descriptive comment explaining the constant's purpose.
