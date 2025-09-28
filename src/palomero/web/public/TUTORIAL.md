# Palomero Web App Tutorial

Welcome to the Palomero Web App! This tutorial will guide you through the
features and functionalities of the application, helping you align images and
transfer ROIs using a user-friendly interface.

---

## Introduction

The Palomero Web App provides a graphical interface for the powerful image
alignment and ROI transfer capabilities of the [`palomero` command-line
tool](https://github.com/yu-anchen/palomero). It allows you to manage different
projects, run alignment tasks on pairs or batches of images from your OMERO
server, and view the results and quality control (QC) images directly in your
browser.

---

## Getting Started

Before launching the web app, you need to be logged into your OMERO server. You
can do this by running the following command in your terminal, which will prompt
you for your credentials:

```bash
# Log in to your OMERO server (replace placeholders)
# The -t flag sets session timeout in seconds (optional but recommended for long runs)
omero login -s <your.omero.server> -u <username> -p <port> -t 999999
```

More informations for the login command can be found
[here](https://omero.readthedocs.io/en/stable/users/cli/sessions.html#login).

Once you are logged in, you can launch the Palomero Web App:

```bash
palomero-web
```

This will start a local web server, and you can access the application by
navigating to `http://localhost:5001` in your web browser.

The footer of the web app displays the status of the OMERO connection. A ⚡️ icon
indicates a successful connection, while a ⛔ icon means the app cannot connect
to the OMERO server.

<!-- Screenshot of the main page with the OMERO connection status highlighted. -->

---

## Projects

Projects allow you to organize your alignment tasks. Each project has a name and
a description, and contains a set of related tasks.

### Creating a New Project

1. Click on the "New" button in the "Project" section of the navigation bar.
2. Fill in the "Name" and "Description" for your new project.
3. Click "Create".

You will be redirected to the new project's page, where you can start adding
alignment tasks.

<!-- Screenshot of the new project form. -->

### Switching Between Projects

You can switch between existing projects by clicking on the project name
dropdown in the navigation bar and selecting the desired project.

<!-- Screenshot of the project dropdown menu. -->

### Deleting a Project

To delete a project, including all of its associated tasks:

1. Switch to the project you want to delete.
2. Click the "Delete" button in the "Project" section of the navigation bar.
3. Confirm the deletion in the dialog box that appears.

---

## Running Alignments

You can run alignment tasks in two ways: on a single pair of images or in a
batch using a CSV file.

### Single Pair Alignment

1. From a project page, click on "Pair" under the "Run" section in the left-hand
   sidebar.
2. Fill in the form with the details of the two images you want to align
   (`image-id-from` and `image-id-to`).
3. Adjust the alignment parameters as needed. The default parameters are
   optimized for most use cases.
4. Click "Run Pair" to submit the task.

<!-- Screenshot of the single pair alignment form. -->

### Batch Processing

For running multiple alignment tasks at once, you can use a CSV file.

1. From a project page, click on "Batch" under the "Run" section in the
   left-hand sidebar.
2. Create a CSV file with the required headers: `image-id-from` and
   `image-id-to`. You can also include other parameters as columns to override
   the default settings for specific pairs.
3. Click the "Choose File" button and select your CSV file. The app will
   validate the file and display a preview of the tasks to be run.
4. Adjust any global alignment parameters in the form.
5. Click "Run" to submit the batch of tasks.

<!-- Screenshot of the batch processing form with a CSV file loaded. -->

---

## Viewing Results

Once you have submitted tasks, you can monitor their progress and view the
results.

### Task Status

The "Results" section in the left-hand sidebar lists all the tasks for the
current project. The status of each task is indicated by an icon:

- ⏸️: The task is queued and waiting to be processed.
- <span aria-busy="true"></span>: The task is currently being processed.
- ✅: The task has completed successfully.

### Logs and QC Images

Click on a task in the "Results" list to view its details. On the task page, you
can find:

- **Run Settings**: A summary of the parameters used for the alignment.
- **Log**: The detailed log output from the `palomero` command-line tool. This
  is useful for debugging any issues.
- **QC Images**: After a task completes, quality control images will be
  displayed, showing the results of the alignment. You can click on these images
  to view them in full size.

<!-- Screenshot of a completed task page, showing the logs and QC images. -->