<h1 align = "center">HOWTO</h1>

<div align = "justify">

**Objective:** A simple *step-by-step* approach on how to run the code, and get model analysis. This file is developed such that any *new contributor* is at par
with the coding style and quickly start the analysis. Warning: the documentation maybe incomplete/haphazard. Consider the following steps:

1. First run `[INTERNAL CODE] Processing RAW Data into Actionable AI-ML Formats.ipynb` file using the raw excel file format.
   * the raw file is always ignored from the repository.
   * data file is generated seperately.
   * data file is downloaded using [`dataIO`](https://github.com/neuralNOD/dataIO) repository, and processed.
2. Run the `trainer.py` using any defined models under `./src/models` and the model file is saved under `savedmodels` directory.
   * again the saved models is ignored from commits.
   * do not attempt to alter/commit the trained models file - else any pull request will be reejcted.
3. Finally, evaluate the model using evaluation notebook.

</div>