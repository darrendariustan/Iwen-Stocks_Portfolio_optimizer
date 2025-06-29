# Deploying Portfolio Optimizer to Render

This guide explains how to deploy the Stock Portfolio Optimizer application to Render using Docker.

## Deployment Steps

### 1. Prepare your GitHub repository

Make sure your repository includes:
- The application code (`portfolio_optimizer.py` and other Python files)
- `requirements.txt` with all dependencies
- `runtime.txt` specifying Python version
- `Dockerfile` and `.dockerignore`

### 2. Create a new Web Service on Render

1. Log in to [Render](https://render.com/)
2. Click on "New +" in the dashboard
3. Select "Web Service"
4. Connect your GitHub repository

### 3. Configure the Web Service

1. **Name**: Choose a name for your application (e.g., "stock-portfolio-optimizer")
2. **Environment**: Select "Docker"
   - Render will automatically detect your Dockerfile
3. **Region**: Choose the data center closest to your target users
4. **Branch**: Select the branch you want to deploy (usually "main" or "master")
5. **Build Command**: Leave empty (handled by Dockerfile)
6. **Start Command**: Leave empty (handled by Dockerfile)

### 4. Configure Environment Variables

You may need to add the following environment variables:
- `PORT`: `8501` (Streamlit's default port)
- Add any API keys or sensitive information that your app needs

### 5. Choose a Plan

Select an appropriate plan based on your application's resource needs:
- For testing: Free plan
- For production: Paid plan with appropriate resources

### 6. Click "Create Web Service"

Render will build your Docker image and deploy your application. This process may take a few minutes.

### 7. Access Your Deployed Application

Once the build and deployment are complete, you can access your application at the URL provided by Render (usually `https://your-app-name.onrender.com`).

## Troubleshooting

If you encounter any issues:

1. Check the build logs on Render for errors
2. Verify that all dependencies are correctly specified in `requirements.txt`
3. Ensure the Dockerfile is correctly configured
4. Check if any environment variables are missing

## Updating Your Application

To update your application after making changes:

1. Push changes to your GitHub repository
2. Render will automatically rebuild and deploy the updated application

## Local Testing Before Deployment

To test your Dockerized application locally before deployment:

```bash
# Build the Docker image
docker build -t portfolio-optimizer .

# Run the container
docker run -p 8501:8501 portfolio-optimizer
```

Then access the application at `http://localhost:8501`
