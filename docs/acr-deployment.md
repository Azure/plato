# Deploying Agent Using ACR

## Prerequisites

Before getting started with agent deployment using ACR, you need access to a docker license and write permissions to the Azure Container Registry in Azure. 


### Create a Resource Group
To create a resource group, use the _az group create_ command. 

```
az group create --name $resourcegroup --location $location
```



### Create a Container Registry
To create a container registry in ACR, you will use the _az acr create_ command as so:

```
az acr create --resource-group $resourcegroup --name $containerregistry --sku $acrtier
```

There are three ACR tiers that you can choose from- Basic, Standard and Premium. See Azure's [Container Registry Service Tiers](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-skus) for more details.



### Login and Push Docker to ACR

Use the login command below to pull and push container images to ACR:

```
az acr login --subscription $subscriptionid --name $acrname
```

Build, tag and push your docker container to ACR

```
docker build -t $imagename -f Dockerfile 
docker tag $imagename $acrname.azurecr.io/$imagename
docker push $acrname.azurecr.io/$imagename
```




### ACR Registry Authentication

Once you have created a container registry and resource group, you will need to establish your registry authentication. For the purposes of agent deployment, it can be advantageous to set up a service principal for __headless__ service. Headless services allow for you to automat push or pull commands of container images for container orchestratation if you are using a service such as Kubernetes and CI/CD solutions when using a service such as Jenkins.

To create a service principal that has access to your Azure resources, use the az ad sp create-for-rbac command as below:

```
az ad sp create-for-rbac --name $serviceprincipal --scopes $acrregistryid --role $acrrole
```

See Azure's overview on [Service Principals](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-auth-service-principal) for more details.



### Store Credentials in Key Vault
Once you've created a service principal, you can save the principal in Azure's Key Vault for safe keeping using the _az keyvault secret set_ command.

To store your service principal password, use this command:

```
az keyvault secret set --vault-name $akvname --name $acrname-pull-pwd --value (az ad sp show --id $principalappId --query password --output tsv)
```

To store your service principal username, use this command:

```
az keyvault secret set --vault-name $akvname --name $acrname-pull-usr --value (az ad sp show --id $principalappId --query password --output tsv)
```


### Deploy Container 

Using the _az conrainter create_ command, you can deploy your container instance as below:

```
# Set dns-name-label to random for a unique label value. dns-name-label must be unique within Azure
# Final query command outputs the containers fully qualified domain name (FQDN) in command line. You can paste the FQDN into your browser to verify the application is running 

az container create \
    --name $containername \
    --resource-group $resourcegroup \
    --image $acrname.azurecr.io/$imagename \
    --registry-login-server $acrname.azurecr.io \
    --registry-username $(az keyvault secret show --vault-name $akvname -n $acrname-pull-usr --query value -o tsv) \
    --registry-password $(az keyvault secret show --vault-name $akvname -n $acrname-pull-pwd --query value -o tsv) \
    --dns-name-label aci-demo-$RANDOM \
    --query ipAddress.fqdn

```

 

## ACR Managed Identity Steps
If your Azure Container Registry's Public Access is set to 'Select Networks' or 'None', you will be usin g Managed Identity based authentication with ACR. The Instructions on how to do so is below.

A managed idenetity for Azure resources allows you assign system or user access to ACR over VMs.

Follow the following steps first
- Create a Resource Group
- Create a Container Registry
- Login to Registry

Once you've completed these tasks, you can begin following these steps.




## Create a Docker-enabled VM
If you do not have an Azure virtual machine already, you can create one using the _az vm create_ command. 

```
az vm create \
    --resource-group $resourcegroup \
    --name $dockername \
    --image $vmtype \
    --admin-username $username \
    --generate-ssh-keys
```

For more information on creating VMs, follow [Azure's az vm command glossary](https://learn.microsoft.com/en-us/cli/azure/vm?view=azure-cli-latest#az_vm_create)

After you have created a Docker-enabled VM, you must install Docker on your VM. Follow these commands to do so:

```
ssh azureuser@$ipaddressofvm
sudo apy update
sudo apt install docker.io -y
```

To verify that Docker is properly installed, you can run Microsoft's 'hello-world' demo using this command:

```
sudo docker run -it mcr.microsoft.com/hello-world
```

After this, you can grant access with a user-assigned identity or system-assigned identity. See Azure's [Use an Azure managed identity to authenticate to an Azure container registry](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication-managed-identity?tabs=azure-cli) documentation for more details. 

