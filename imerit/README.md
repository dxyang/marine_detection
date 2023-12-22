## Instructions for using AWS CLI

- Install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

## Downloading data from AWS:
/usr/local/bin/aws configure sso
SSO start URL: <example from invite email: https://d-90679b6642.awsapps.com/start/>
SSO region: us-east-1

This will generate a profile name, save this for subsequent commands.

Then to list data:
/usr/local/bin/aws s3 ls s3://whoi-rsi-fish-detection/datasets/ --profile <profile name>

Or to download data:
/usr/local/bin/aws s3 sync s3://whoi-rsi-fish-detection/datasets/ <local destination folder> --profile <profile name>
