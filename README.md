# How to use TensorFlow with Google VertexAI

Hello everyone, today I will discuss how to deploy a training pipeline which uses TensorFlow on Vertex AI and deploy an endpoint for the model in the cloud using Vertex AI for online prediction.

The dataset used for this tutorial is the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist) from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/overview). The version of the dataset you will use is built into TensorFlow. The trained model predicts which type of class (digit) an image is from ten classes (0-9)

The steps performed include:

- Create a Vertex AI custom job for training a model in distributed fashion.
- Train the model using TensorFlow's `MirroredStrategy`.
- Deploy the `Model` resource to a serving `Endpoint` resource.
- Make a prediction.
- Undeploy the `Model` resource.

Let us  create a custom-trained model from a Python script in a Docker container using the Vertex SDK for Python, and then do a prediction on the deployed model by sending data. 

1.   Train a model using distribution strategies on Vertex AI using the SDK for Python
2.   Deploy a custom image classification model for online prediction using Vertex AI

## Step 1 - Activate Cloud Shell

First we login into our console of  Google Cloud Platform (GCP) account 

[https://cloud.google.com](https://cloud.google.com)

![](assets/images/posts/README/login.jpg)

Then after you login to your Cloud Console  in the top right toolbar, click the **Activate Cloud Shell** button.

![](assets/images/posts/README/1.jpg)

Click **Continue**.

![](assets/images/posts/README/2.jpg)

It takes a few moments to provision and connect to the environment. When you are connected, you are already authenticated, and the project is set to your *PROJECT_ID*. For example:

![](assets/images/posts/README/3.jpg)

`gcloud` is the command-line tool for Google Cloud. It comes pre-installed on Cloud Shell and supports tab-completion.

You can list the active account name with this command:

```
gcloud auth list
```

then you should **authorize**

![](assets/images/posts/README/4.jpg)

you will have something like

```
ACTIVE: *
ACCOUNT: username-xxxxxxxxxxxx@yourdomain.com
To set the active account, run:
    $ gcloud config set account `ACCOUNT`
```

You can list the project ID with this command:

```
gcloud config list project
```

for example the output may be similar like

```
[core]
project = <project_ID>
```

## Step 2 Enable Google Cloud services

1. In Cloud Shell, use `gcloud` to enable the services used in the lab

```
gcloud services enable \
  compute.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  notebooks.googleapis.com \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  container.googleapis.com
```

You will obtain something like

```
Operation "operations/acf.p2-258289527804-d749b4a5-9esb2-4f84-a549-3b6274273ba1" finished successfully.
```



2. Create a custom service account

```
SERVICE_ACCOUNT_ID=vertex-custom-training-sa
gcloud iam service-accounts create $SERVICE_ACCOUNT_ID  \
    --description="A custom service account for Vertex custom training" \
    --display-name="Vertex AI Custom Training"
```

3. Set the **Project ID** environment variable

```
PROJECT_ID=$(gcloud config get-value core/project)
```

you will get

```
Your active configuration is: [cloudshell-2853]
```

4. Grant your service account the `aiplatform.user` role.

```
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$SERVICE_ACCOUNT_ID@$PROJECT_ID.iam.gserviceaccount.com \
    --role="roles/aiplatform.user"
```

This will allow access to running model training, deployment, and explanation jobs with Vertex AI.

## Step 3 Deploy Vertex Notebook instance

To launch Notebooks with Vertex AI:

1. Click on the **Navigation Menu** and navigate to **Vertex AI**, then to **Workbench**.

   ![](assets/images/posts/README/5.jpg)

   then we activate the API

   ![](assets/images/posts/README/6.jpg)

2. On the Notebook instances page, click **New Notebook**.

3. In the **Customize instance** menu, select **TensorFlow Enterprise** and choose the latest version of **TensorFlow Enterprise 2.x (with LTS)** > **Without GPUs**.

![](assets/images/posts/README/7.jpg)

1. In the **New notebook instance** dialog, click the pencil icon to **Edit** instance properties.
2. For **Instance name**, enter a name for your instance.
3. For **Region**, select `us-central1` and for **Zone**, select a zone within the selected region.

![](assets/images/posts/README/8.jpg)

4.Then we click on 4. **Advanced options** and scroll down to Machine configuration and select **n1-standard-2** for Machine type.

5. Leave the remaining fields with their default and click **Create**.

![](assets/images/posts/README/9.jpg)



after create you will see

![](assets/images/posts/README/10.jpg)

After a few minutes, the Vertex AI console will display your instance name, followed by **Open JupyterLab**.

![](assets/images/posts/README/11.jpg)

Click **Open JupyterLab**. A JupyterLab window will open in a new tab.

![](assets/images/posts/README/12.jpg)



Let us create a new notebook

Install the latest (preview) version of Vertex SDK for Python.


```python
import os
```


```python
!pip3 install --user --upgrade google-cloud-aiplatform
```


Install the latest GA version of *google-cloud-storage* library as well.


```python
!pip3 install --user --upgrade google-cloud-storage
```


Install the *pillow* library for loading images.


```python
!pip3 install --user --upgrade pillow
```

    Requirement already satisfied: pillow in /opt/conda/lib/python3.7/site-packages (9.1.1)


Install the *numpy* library for manipulation of image data.


```python
! pip3 install --user --upgrade numpy
```

    Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (1.21.6)


You can safely ignore errors during the numpy installation.

### Restart the kernel

Once you've installed everything, you need to restart the notebook kernel so it can find the packages.


```python
import os

if not os.getenv("IS_TESTING"):
    # Automatically restart kernel after installs
    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
```

#### Set your project ID

**If you don't know your project ID**, you may be able to get your project ID using `gcloud`. 


```python
import os

PROJECT_ID = ""

if not os.getenv("IS_TESTING"):
    # Get your Google Cloud project ID from gcloud
    shell_output=!gcloud config list --format 'value(core.project)' 2>/dev/null
    PROJECT_ID = shell_output[0]
    print("Project ID: ", PROJECT_ID)
```

    Project ID:  yourproject-xxxx-id


#### Timestamp

If you are in a live tutorial session, you might be using a shared test account or project. To avoid name collisions between users on resources created, you create a timestamp for each instance session, and append it onto the name of resources you create in this tutorial.


```python
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
```

### Create a Cloud Storage bucket

**The following steps are required, regardless of your notebook environment.**

When you submit a training job using the Cloud SDK, you upload a Python package
containing your training code to a Cloud Storage bucket. Vertex AI runs
the code from this package. In this tutorial, Vertex AI also saves the
trained model that results from your job in the same bucket. Using this model artifact, you can then
create Vertex AI model and endpoint resources in order to serve
online predictions.

Set the name of your Cloud Storage bucket below. It must be unique across all
Cloud Storage buckets.

You may also change the `REGION` variable, which is used for operations
throughout the rest of this notebook. Make sure to [choose a region where Vertex AI services are available](https://cloud.google.com/vertex-ai/docs/general/locations#available_regions). You may
not use a Multi-Regional Storage bucket for training with Vertex AI.


```python
BUCKET_NAME = "gs://test_bucket_ruslanmv1230"
REGION = "us-central1"  # @param {type:"string"}
```


```python
if BUCKET_NAME == "" or BUCKET_NAME is None or BUCKET_NAME == "gs://test_bucket_ruslanmv1230":
    BUCKET_NAME = "gs://" + PROJECT_ID
```

**Only if your bucket doesn't already exist**: Run the following cells to create your Cloud Storage bucket.


```python
! gsutil mb -l $REGION $BUCKET_NAME
```


### Set up variables

Next, set up some variables used throughout the tutorial.

#### Import Vertex SDK for Python

Import the Vertex SDK for Python into your Python environment and initialize it.


```python
import os
import sys

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)
```

#### Set hardware accelerators

Here to run a container image on a CPU, we set the variables `TRAIN_GPU/TRAIN_NGPU` and `DEPLOY_GPU/DEPLOY_NGPU` to `(None, None)` since this notebook is meant to be run in a Qwiklab environment where GPUs cannot be provisioned. 

Note: If you happen to be running this notebook from your personal GCP account, set the variables `TRAIN_GPU/TRAIN_NGPU` and `DEPLOY_GPU/DEPLOY_NGPU` to use a container image supporting a GPU and the number of GPUs allocated to the virtual machine (VM) instance. For example, to use a GPU container image with 4 Nvidia Tesla K80 GPUs allocated to each VM, you would specify:

    (aip.AcceleratorType.NVIDIA_TESLA_K80, 4)

See the [locations where accelerators are available](https://cloud.google.com/vertex-ai/docs/general/locations#accelerators).


```python
TRAIN_GPU, TRAIN_NGPU = (None, None)
DEPLOY_GPU, DEPLOY_NGPU = (None, None)
```

#### Set pre-built containers

Vertex AI provides pre-built containers to run training and prediction.

For the latest list, see [Pre-built containers for training](https://cloud.google.com/vertex-ai/docs/training/pre-built-containers) and [Pre-built containers for prediction](https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers)


```python
TRAIN_VERSION = "tf-cpu.2-6"
DEPLOY_VERSION = "tf2-cpu.2-6"

TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/{}:latest".format(TRAIN_VERSION)
DEPLOY_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/{}:latest".format(DEPLOY_VERSION)

print("Training:", TRAIN_IMAGE, TRAIN_GPU, TRAIN_NGPU)
print("Deployment:", DEPLOY_IMAGE, DEPLOY_GPU, DEPLOY_NGPU)
```

    Training: us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-6:latest None None
    Deployment: us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest None None


#### Set machine types

Next, set the machine types to use for training and prediction.

- Set the variables `TRAIN_COMPUTE` and `DEPLOY_COMPUTE` to configure your compute resources for training and prediction.
 - `machine type`
   - `n1-standard`: 3.75GB of memory per vCPU
   - `n1-highmem`: 6.5GB of memory per vCPU
   - `n1-highcpu`: 0.9 GB of memory per vCPU
 - `vCPUs`: number of \[2, 4, 8, 16, 32, 64, 96 \]

*Note: The following is not supported for training:*

 - `standard`: 2 vCPUs
 - `highcpu`: 2, 4 and 8 vCPUs

*Note: You may also use n2 and e2 machine types for training and deployment, but they do not support GPUs*.


```python
MACHINE_TYPE = "n1-standard"

VCPU = "4"
TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Train machine type", TRAIN_COMPUTE)

MACHINE_TYPE = "n1-standard"

VCPU = "4"
DEPLOY_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Deploy machine type", DEPLOY_COMPUTE)
```

    Train machine type n1-standard-4
    Deploy machine type n1-standard-4


# Distributed training and deployment

Now you are ready to start creating your own custom-trained model with MNIST and deploying it as online prediction service.

## Train a model

There are two ways you can train a custom model using a container image:

- **Use a Google Cloud prebuilt container**. If you use a prebuilt container, you will additionally specify a Python package to install into the container image. This Python package contains your code for training a custom model.

- **Use your own custom container image**. If you use your own container, the container needs to contain your code for training a custom model.

### Define the command args for the training script

Prepare the command-line arguments to pass to your training script.

- `args`: The command line arguments to pass to the corresponding Python module. In this example, they will be:
  - `"--epochs=" + EPOCHS`: The number of epochs for training.
  - `"--steps=" + STEPS`: The number of steps (batches) per epoch.
  - `"--distribute=" + TRAIN_STRATEGY"` : The training distribution strategy to use for single or distributed training.
    - `"single"`: single device.
    - `"mirror"`: all GPU devices on a single compute instance.
    - `"multi"`: all GPU devices on all compute instances.


```python
JOB_NAME = "custom_job_" + TIMESTAMP
MODEL_DIR = "{}/{}".format(BUCKET_NAME, JOB_NAME)

if not TRAIN_NGPU or TRAIN_NGPU < 2:
    TRAIN_STRATEGY = "single"
else:
    TRAIN_STRATEGY = "mirror"

EPOCHS = 20
STEPS = 100

CMDARGS = [
    "--epochs=" + str(EPOCHS),
    "--steps=" + str(STEPS),
    "--distribute=" + TRAIN_STRATEGY,
]
```


```python
TRAIN_STRATEGY
```


    'single'

#### Training script

In the next cell, you will write the contents of the training script, `task.py`. In summary:

- Get the directory where to save the model artifacts from the environment variable `AIP_MODEL_DIR`. This variable is set by the training service.
- Loads MNIST dataset from TF Datasets (tfds).
- Builds a model using TF.Keras model API.
- Compiles the model (`compile()`).
- Sets a training distribution strategy according to the argument `args.distribute`.
- Trains the model (`fit()`) with epochs and steps according to the arguments `args.epochs` and `args.steps`
- Saves the trained model (`save(MODEL_DIR)`) to the specified model directory.


```python
%%writefile task.py
# Single, Mirror and Multi-Machine Distributed Training for MNIST

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import os
import sys
tfds.disable_progress_bar()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', dest='lr',
                    default=0.01, type=float,
                    help='Learning rate.')
parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')
parser.add_argument('--steps', dest='steps',
                    default=200, type=int,
                    help='Number of steps per epoch.')
parser.add_argument('--distribute', dest='distribute', type=str, default='single',
                    help='distributed training strategy')
args = parser.parse_args()

print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())

# Single Machine, single compute device
if args.distribute == 'single':
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
# Single Machine, multiple compute device
elif args.distribute == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
# Multiple Machine, multiple compute device
elif args.distribute == 'multi':
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Multi-worker configuration
print('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))

# Preparing dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64

def make_datasets_unbatched():
  # Scaling MNIST data from (0, 255] to (0., 1.]
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label

  datasets, info = tfds.load(name='mnist',
                            with_info=True,
                            as_supervised=True)
  return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE).repeat()


# Build the Keras model
def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr),
      metrics=['accuracy'])
  return model

# Train the model
NUM_WORKERS = strategy.num_replicas_in_sync
# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size.
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
MODEL_DIR = os.getenv("AIP_MODEL_DIR")

train_dataset = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)

with strategy.scope():
  # Creation of dataset, and model building/compiling need to be within
  # `strategy.scope()`.
  model = build_and_compile_cnn_model()

model.fit(x=train_dataset, epochs=args.epochs, steps_per_epoch=args.steps)
model.save(MODEL_DIR)
```

    Writing task.py


### Train the model

Define your custom training job on Vertex AI.

Use the `CustomTrainingJob` class to define the job, which takes the following parameters:

- `display_name`: The user-defined name of this training pipeline.
- `script_path`: The local path to the training script.
- `container_uri`: The URI of the training container image.
- `requirements`: The list of Python package dependencies of the script.
- `model_serving_container_image_uri`: The URI of a container that can serve predictions for your model — either a prebuilt container or a custom container.

Use the `run` function to start training, which takes the following parameters:

- `args`: The command line arguments to be passed to the Python script.
- `replica_count`: The number of worker replicas.
- `model_display_name`: The display name of the `Model` if the script produces a managed `Model`.
- `machine_type`: The type of machine to use for training.
- `accelerator_type`: The hardware accelerator type.
- `accelerator_count`: The number of accelerators to attach to a worker replica.

The `run` function creates a training pipeline that trains and creates a `Model` object. After the training pipeline completes, the `run` function returns the `Model` object.

You can read more about the `CustomTrainingJob.run` API [here](https://googleapis.dev/python/aiplatform/latest/aiplatform.html?highlight=customtraining#google.cloud.aiplatform.CustomTrainingJob.run)


```python
job = aiplatform.CustomTrainingJob(
    display_name=JOB_NAME,
    script_path="task.py",
    container_uri=TRAIN_IMAGE,
    requirements=["tensorflow_datasets==1.3.0"],
    model_serving_container_image_uri=DEPLOY_IMAGE,
)

MODEL_DISPLAY_NAME = "mnist-" + TIMESTAMP

# Start the training
if TRAIN_GPU:
    model = job.run(
        model_display_name=MODEL_DISPLAY_NAME,
        args=CMDARGS,
        replica_count=1,
        machine_type=TRAIN_COMPUTE,
        accelerator_type=TRAIN_GPU.name,
        accelerator_count=TRAIN_NGPU,
    )
else:
    model = job.run(
        model_display_name=MODEL_DISPLAY_NAME,
        args=CMDARGS,
        replica_count=1,
        machine_type=TRAIN_COMPUTE,
        accelerator_count=0,
    )
```




To view the training pipeline status, you have to navigate to **Vertex AI** ➞ **Training**  

<img src="assets/images/posts/README/vertex-ai-training.png" width="200">

You can see the status of the current training pipeline as seen below

![](assets/images/posts/README/14.jpg)

Once the model has been successfully trained, you can see a custom trained model if you head to **Vertex AI** ➞ **Models**

<img src="assets/images/posts/README/screenshot-2021-08-19-at-7-41-25-pm.png" width="200">





![](assets/images/posts/README/15.jpg)

### Deploy the model

Before you use your model to make predictions, you need to deploy it to an `Endpoint`. You can do this by calling the `deploy` function on the `Model` resource. This will do two things:

1. Create an `Endpoint` resource for deploying the `Model` resource to.
2. Deploy the `Model` resource to the `Endpoint` resource.


The function takes the following parameters:

- `deployed_model_display_name`: A human readable name for the deployed model.
- `traffic_split`: Percent of traffic at the endpoint that goes to this model, which is specified as a dictionary of one or more key/value pairs.
  - If only one model, then specify as **{ "0": 100 }**, where "0" refers to this model being uploaded and 100 means 100% of the traffic.
  - If there are existing models on the endpoint, for which the traffic will be split, then use `model_id` to specify as **{ "0": percent, model_id: percent, ... }**, where `model_id` is the model id of an existing model to the deployed endpoint. The percents must add up to 100.
- `machine_type`: The type of machine to use for training.
- `accelerator_type`: The hardware accelerator type.
- `accelerator_count`: The number of accelerators to attach to a worker replica.
- `starting_replica_count`: The number of compute instances to initially provision.
- `max_replica_count`: The maximum number of compute instances to scale to. In this tutorial, only one instance is provisioned.

### Traffic split

The `traffic_split` parameter is specified as a Python dictionary. You can deploy more than one instance of your model to an endpoint, and then set the percentage of traffic that goes to each instance.

You can use a traffic split to introduce a new model gradually into production. For example, if you had one existing model in production with 100% of the traffic, you could deploy a new model to the same endpoint, direct 10% of traffic to it, and reduce the original model's traffic to 90%. This allows you to monitor the new model's performance while minimizing the distruption to the majority of users.

### Compute instance scaling

You can specify a single instance (or node) to serve your online prediction requests. This tutorial uses a single node, so the variables `MIN_NODES` and `MAX_NODES` are both set to `1`.

If you want to use multiple nodes to serve your online prediction requests, set `MAX_NODES` to the maximum number of nodes you want to use. Vertex AI autoscales the number of nodes used to serve your predictions, up to the maximum number you set. Refer to the [pricing page](https://cloud.google.com/vertex-ai/pricing#prediction-prices) to understand the costs of autoscaling with multiple nodes.

### Endpoint

The method will block until the model is deployed and eventually return an `Endpoint` object. If this is the first time a model is deployed to the endpoint, it may take a few additional minutes to complete provisioning of resources.


```python
DEPLOYED_NAME = "mnist_deployed-" + TIMESTAMP

TRAFFIC_SPLIT = {"0": 100}

MIN_NODES = 1
MAX_NODES = 1

if DEPLOY_GPU:
    endpoint = model.deploy(
        deployed_model_display_name=DEPLOYED_NAME,
        traffic_split=TRAFFIC_SPLIT,
        machine_type=DEPLOY_COMPUTE,
        accelerator_type=DEPLOY_GPU.name,
        accelerator_count=DEPLOY_NGPU,
        min_replica_count=MIN_NODES,
        max_replica_count=MAX_NODES,
    )
else:
    endpoint = model.deploy(
        deployed_model_display_name=DEPLOYED_NAME,
        traffic_split=TRAFFIC_SPLIT,
        machine_type=DEPLOY_COMPUTE,
        accelerator_type=None,
        accelerator_count=0,
        min_replica_count=MIN_NODES,
        max_replica_count=MAX_NODES,
    )
```

    Creating Endpoint


In order to view your deployed endpoint, you can head over to **Vertex AI** ➞ **Endpoints**

<img src="assets/images/posts/README/vertex-ai-endpoints.png" width="200">



You can check if your endpoint is in the list of the currently deployed/deploying endpoints.



To view the details of the endpoint that is currently deploying, you can simply click on the endpoint name.

Once deployment is successfull, you should be able to see a green tick next to the endpoint name in the above screenshot.

![screenshot-2021-08-19-at-7-06-14-pm.png](data:image/png;base64,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AwAAAFgmjZanniOH0CqR3lauk5PTtGnTTp486enpOX/+fNXld2Ji4pQpU3755RdVBgAAALCM/nAm1VFeTEzMsmXL1JNyrK2tBw8ebBz5XzFdutrASsuWLZcvXx4QEPD333+//fbbhQsXlswLFy7o3YVlFv0UdMOGDbUzzsnMnTtXCug9p/n6+qpEasw/rf0IyWGClkqFBb9apssh5+0zGH1IUWZ9vXphd3d3lcgiuSDUJKSyKFq0qCRkK509e7bKFHq/mWfPni2dChsbGymgb5l79uxRiXTRf9ejR48mibpLTnh4uCRkfdUjhAAAAIBl9Jbn7t27VUKnnjQrmjRpohLmt3LPnz//h9GZM2dksEKFCm+99db27dtVT/pCDtFVAgAAAMigvn375s2bVxLffPPN+vXrJdG+fXsPDw/jyBSkqw28evVqadauWrUqPj7ewcGhS5cus2fP3rdvn7rD4+rVqxcvXlQlM4XpKWhXV1ftpPOD1NOh9ACSNLkTEhJUWlmyZMm7RpcvX5ZB809r52Tp+tWySA45b68fl504cSKzbkfLrK/3+PHjKpHVwcvcEWrKly/fpEmTVPqzzz4LDAxU6W7duqmE1CYXLlxQaeWvv/4aNGhQTEyMGtRLzp8/38/PT6WFFEvyTLAUVaxYsUGDBpIICgoyjXVJlfHBBx+otLotFAAAAMiIfv36qcTcuXMDAgJUWhw+fFjarpKwsrJq3LixyjS/levl5fW80XPPPWd63KtubBL6030BAACADHJxcenTp48kLl68qLqVU/c5pSZdbeDPPvtMNWtXrFihckTFihX1R9roLVsHBweViIyMtDgAIJNV98TIRN555x2Vqcg0+/fvv2PHDjVYpUqVChUqSELa3itXrlSZ4vbt22+++eYXX3zxzTfflChRQnLMP62dLpmyvOZL16+WcfrSmX5WDjlvL7971apVJRESEiI/tMoUUVFRq1ev1gbSybKv9++//1YXFyp79uzZu3evJGxtbVu1aqUys0juCDWJYcOGlSpVShJyzPzRRx+pzF69eqnNMiIiQr7WDz/8cMuWLb/++uvrr7/eo0ePX375ZejQoapk3759q1WrJgnZsJs1azZnzpw//vhjzJgxUiXpHSmmcUug/GzqLkgxdepUqRo2bNggdZl8+p9//imZsjJJvioAAAAApGHatGlvp0SasjK2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371798KFC1+/fr1Zs2bt27eXt9euXdvFxeXGjRuJiYmqmCKZHY1at27t4eERFRWVkJAgE5R0ktlITj5U3igTr1mzZpEiRWTi8nF58+aVOQkICNAKyS9qY9OoUaNu3bpJYSlZokQJmf/o6Ght9P0ZljmR/A4dOsgMy6t8UXZ2drdu3dIKGcknyld05cqVq1evall58tja2jZs2PCZZ555+umn69atW7p06YiIiJCQEDXW2dm5f//+8kZfX9/w8HCVqfTq1atx48ayLulz+9BZVcxccDO1bdtWSwEAAAAAAAAAgAx7QkNN1tbWnTt3zps377///msarihTpsyIESPc3d0dHR1tbW1Lly5dq1atwoULnzt3TgWNXF1de/bsWbRo0a1bt6q3lC1btn379g4ODlWrVm3dunWRIkVksgUKFJCPlrcfP35cFROlSpV64403KleuLBOR6ZcsWbJ+/fqSrl27tgzu27dPK5eMlZVV9+7du3Xr5ubmJhOXt1SoUKFmzZrOzs729vamERcp+eKLLz711FOFChVKSEgoVqyYh4eHfMqNGzfu3r2ryqgZzp8/f9OmTWXpZGoyHSlZo0aNggULnj9/Xg+PJQ81yfQHDhzYtm1bmb6kZTHlu2rYsKGMunbtmrzKqtKuXTuZvcjISHmj8U0GMuX+/fvLp2zZsiUiIkJyzJlV8xfcfISaAAAAAAAAAADIRDk91HT8csTivwMX/hnww/qA0vantVyzJQ81WVlZlSlTpmvXrpIfHBy8bt26hIQENapQoUJvvPGGvb39n3/++euvv27fvv3y5ctVqlQpW7aslFR3/OTLl6958+aS0ENNJUqUUAGbvHnzrjI6ePBgdHS0TL9IkSJXrlyR90ox+dzXX3+9cOHCt2/f/t///icfcfToUfnoJk2ayNjw8PA0Qk0y/e7duycmJm7cuHHp0qXbtm27evWqZLq4uMhY04jLM888IxP08vJauHChFN61a5eTk1MFo/3796sYkprh/Pnzy4fKYq5du3bv3r1xcXEywyVLlpRJ6XdxJQ819erVq379+lJm0aJF8tXt3LlTlk6+okqVKt25c8fX11fKODg4VK5cWaZvukQNGzaUYj4+PvKtqhxzZtX8BTcfoSYAAAAAAAAAADJRjn5W0/HLEfNX3d598t6d0DgtyyJlypQZed+UKVPktV69ehcvXvz6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)

## Make an online prediction request

Send an online prediction request to your deployed model.

### Testing

Get the test dataset and load the images/labels.

Set the batch size to -1 to load the entire dataset.


```python
import tensorflow_datasets as tfds
import numpy as np
```


```python
datasets, info = tfds.load(name='mnist',
                            with_info=True,
                            batch_size=-1,
                            as_supervised=True)

test_dataset = datasets['test']
```

Load the TensorFlow Dataset as NumPy arrays (images, labels)


```python
x_test, y_test = tfds.as_numpy(test_dataset)

# Normalize (rescale) the pixel data by dividing each pixel by 255. 
x_test = x_test.astype('float32') / 255.
```

Ensure the shapes are correct here


```python
x_test.shape, y_test.shape
```


```python
#@title Pick the number of test images
NUM_TEST_IMAGES = 20 #@param {type:"slider", min:1, max:20, step:1}
x_test, y_test = x_test[:NUM_TEST_IMAGES], y_test[:NUM_TEST_IMAGES]
```

### Send the prediction request

Now that you have test images, you can use them to send a prediction request. Use the `Endpoint` object's `predict` function, which takes the following parameters:

- `instances`: A list of image instances. According to your custom model, each image instance should be a 3-dimensional matrix of floats. This was prepared in the previous step.

The `predict` function returns a list, where each element in the list corresponds to the corresponding image in the request. You will see in the output for each prediction:

- Confidence level for the prediction (`predictions`), between 0 and 1, for each of the ten classes.

You can then run a quick evaluation on the prediction results:

1. `np.argmax`: Convert each list of confidence levels to a label
2. Compare the predicted labels to the actual labels
3. Calculate `accuracy` as `correct/total`


```python
predictions = endpoint.predict(instances=x_test.tolist())
y_predicted = np.argmax(predictions.predictions, axis=1)

correct = sum(y_predicted == np.array(y_test.tolist()))
accuracy = len(y_predicted)
print(
    f"Correct predictions = {correct}, Total predictions = {accuracy}, Accuracy = {correct/accuracy}"
)
```

## Undeploy the model

To undeploy your `Model` resource from the serving `Endpoint` resource, use the endpoint's `undeploy` method with the following parameter:

- `deployed_model_id`: The model deployment identifier returned by the endpoint service when the `Model` resource was deployed. You can retrieve the deployed models using the endpoint's `deployed_models` property.

Since this is the only deployed model on the `Endpoint` resource, you can omit `traffic_split`.


```python
deployed_model_id = endpoint.list_models()[0].id
endpoint.undeploy(deployed_model_id=deployed_model_id)
```

# Cleaning up

To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.

Otherwise, you can delete the individual resources you created in this tutorial:

- Training Job
- Model
- Endpoint
- Cloud Storage Bucket


```python
delete_training_job = True
delete_model = True
delete_endpoint = True

# Warning: Setting this to true will delete everything in your bucket
delete_bucket = True

# Delete the training job
job.delete()

# Delete the model
model.delete()

# Delete the endpoint
endpoint.delete()

if delete_bucket and "BUCKET_NAME" in globals():
    ! gsutil -m rm -r $BUCKET_NAME
```



Your can download this  `How-to-use-TensorFlow-with-VertexAI` notebook in your JupyterLab instance.

1. In JupyterLab, click the **Terminal** icon to open a new terminal.
2. To clone the training-data-analyst Github repository, type in the following command, and press **Enter**.

```
git clone https://github.com/ruslanmv/How-to-use-TensorFlow-with-VertexAI.git
```

In your notebook, navigate to this folder:

```
How-to-use-TensorFlow-with-VertexAI/self-paced-labs/vertex-ai/vertex-distributed-tensorflow
```



![](assets/images/posts/README/13.jpg)

Continue the lab in the notebook, and run each cell by clicking the **Run** ( ![run-button.png](assets/images/posts/README/vrDGIoOBYHb7rndaVAJo%252BiWClGQ2qew1JY2pAu5r%252BCQ%253D.png)) icon at the top of the screen. Alternatively, you can execute the code in a cell with **SHIFT + ENTER**.

## Congratulations!

Congratulations! In this lab, you walked through a machine learning experimentation workflow using TensorFlow's distribution strategies and Vertex AI's machine learning services to train and deploy a TensorFlow model to classify images from the CIFAR-10 dataset.
