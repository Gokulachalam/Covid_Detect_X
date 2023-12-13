# Covid_Detect_X


## Overview

This project is a web application for classifying X-ray images into two categories: 'covid19' and 'normal'. It uses a deep learning model implemented with PyTorch and a Flask web framework to provide a user-friendly interface for image classification.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Features

- Upload X-ray images for classification
- Grayscale check to ensure images are likely X-rays
- Real-time classification results
- Selected extensions will be allowd - jpg,jpeg and png
- quicker results
- Accurate x-ray daignosis

## Getting Started

### Prerequisites

Ensure you have the following installed on your machine:

- Python (version 3.x)
- pip package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Gokulachalam/Covid_Detect_X


## Setting Up a Remote Repository 

If you haven't set up a remote repository yet and wish to keep a backup or collaborate with others, you can follow these steps:

1. **Create a Remote Repository:**
   - Go to a code hosting platform such as GitHub, GitLab, or Bitbucket.
   - Create a new repository on the platform.

2. **Get Repository URL:**
   - Copy the URL of your newly created remote repository.

3. **Add Remote Repository in Terminal:**
   - Open your terminal and navigate to your project directory.
   - Use the following command to add a remote repository (let's name it "origin"):

     ```bash
     git remote add origin https://github.com/Gokulachalam/Covid_Detect_X
     ```

   Replace `<repository_url>` with the URL of your remote repository.

4. **Verify Remote Repository:**
   - To verify that the remote repository is added successfully, you can use:

     ```bash
     git remote -v
     ```

   It should display the URL of your remote repository.

5. **Push to Remote Repository:**
   - After making changes to your local repository, you can push those changes to the remote repository:

     ```bash
     git push -u origin master
     ```

   Replace "master" with the name of your branch if it's different.

Now, your project is connected to a remote repository, providing a backup and collaboration platform.

Feel free to refer to the documentation of your chosen code hosting platform for more details on repository management and collaboration.



## File Structure

- /model
  - dense_model.pt
- /templates
  - index.html
  - result.html
  - error.html
- app.py
- README.md
- requirements.txt






## Dependencies
Flask

PyTorch

OpenCV

NumPy

Pillow

## Run Scripy

```bash
    python3 app.py
```


## Install requirements using this command

```bash
     pip install - requirements.txt
```

## Demo



