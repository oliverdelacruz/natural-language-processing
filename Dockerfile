# Use an official Python runtime as a parent image
FROM python:3.6

# Set the working directory to app
ENV ENV_NAME natural-language-processing
WORKDIR $ENV_NAME

# Copy the current directory contents into the container at app
COPY . .
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Configure aws login
RUN echo $(aws configure set aws_access_key_id AKIAJ4TBLFWJOR6FGOCQ \
& aws configure set aws_secret_access_key 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 \
& aws configure set default.region us-east-1) 
RUN aws s3 sync s3://oliverdelacruz/$ENV_NAME . --acl public-read 
 
 
# Define environment variable
ENV NAME $ENV_NAME

# Run bash when the container launches
CMD ["/bin/bash"]