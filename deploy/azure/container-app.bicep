// Prepared configuration. No resources are deployed by writing this file.
param location string = resourceGroup().location
param appName string
param environmentId string
param image string
param identityResourceId string
param identityPrincipalId string
param storageAccountUrl string
param artifactContainer string = 'matchmind'
param recordingIds string = '["118575_clip","118575_h1","118575_h2"]'
param provider string = 'local'
param modelEndpoint string = ''
param modelDeployment string = ''
param keyVaultSecretUrl string = ''
param apiTokenSecretUrl string
param registryServer string

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${identityResourceId}': {} } }
  properties: {
    managedEnvironmentId: environmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: { external: true, targetPort: 8000, transport: 'http', allowInsecure: false }
      registries: [{ server: registryServer, identity: identityResourceId }]
      secrets: concat([{ name: 'api-token', keyVaultUrl: apiTokenSecretUrl, identity: identityResourceId }], empty(keyVaultSecretUrl) ? [] : [{ name: 'model-key', keyVaultUrl: keyVaultSecretUrl, identity: identityResourceId }])
    }
    template: {
      containers: [{
        name: 'matchmind'
        image: image
        resources: { cpu: json('0.5'), memory: '1Gi' }
        env: concat([
          { name: 'MATCHMIND_ARTIFACT_STORE', value: 'azure' }
          { name: 'MATCHMIND_BLOB_ACCOUNT_URL', value: storageAccountUrl }
          { name: 'MATCHMIND_BLOB_CONTAINER', value: artifactContainer }
          { name: 'MATCHMIND_RECORDINGS', value: recordingIds }
          { name: 'MATCHMIND_CACHE', value: '/tmp/matchmind-artifacts' }
          { name: 'MATCHMIND_PROVIDER', value: provider }
          { name: 'AZURE_CLIENT_ID', value: reference(identityResourceId, '2023-01-31').clientId }
          { name: 'AZURE_OPENAI_ENDPOINT', value: modelEndpoint }
          { name: 'AZURE_OPENAI_DEPLOYMENT', value: modelDeployment }
          { name: 'MATCHMIND_API_TOKEN', secretRef: 'api-token' }
        ], empty(keyVaultSecretUrl) ? [] : [{ name: 'AZURE_OPENAI_API_KEY', secretRef: 'model-key' }])
        probes: [
          { type: 'Startup', httpGet: { path: '/ready', port: 8000 }, periodSeconds: 5, failureThreshold: 60 }
          { type: 'Readiness', httpGet: { path: '/ready', port: 8000 }, periodSeconds: 10 }
          { type: 'Liveness', httpGet: { path: '/health', port: 8000 }, periodSeconds: 20 }
        ]
      }]
      scale: { minReplicas: 1, maxReplicas: 2, rules: [{ name: 'http', http: { metadata: { concurrentRequests: '10' } } }] }
    }
  }
}
output serviceUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
