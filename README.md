# How to use TensorFlow with Google VertexAI



Hello everyone, today I will discuss how to deploy a training pipeline which uses TensorFlow on Vertex AI and deploy an endpoint for the model in the cloud using Vertex AI for online prediction.

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
