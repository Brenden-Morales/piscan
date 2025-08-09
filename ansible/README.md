# ULTIMATE COMBINED ANSIBLE

```shell
source .venv/bin/activate

# provision devices with dependencies etc
ansible-playbook -i ansible/inventory.yaml ansible/provision.yaml

# deploy application code to all picams and the shoulderpi
ansible-playbook -i ansible/inventory.yaml ansible/deploy.yaml

# clean application code from all picams and the shoulderpi
ansible-playbook -i ansible/inventory.yaml ansible/clean.yaml
```

---

## Structure

This directory deals with provisioning and deploying a raspberry pi 5 called `shoulderpi` as well as
multiple raspberry pi zero 2w's called `picamX`.