# ULTIMATE COMBINED ANSIBLE

```shell
source .venv/bin/activate
# provision devices with dependencies etc
ansible-playbook -i ansible/inventory.yaml ansible/provision.yaml
# application deployment
ansible-playbook -i ansible/inventory.yaml ansible/deploy_shoulder.yaml
ansible-playbook -i ansible/inventory.yaml ansible/deploy_cameras.yaml
```

---

## Structure

This directory deals with provisioning and deploying a raspberry pi 5 called `shoulderpi` as well as
multiple raspberry pi zero 2w's called `picamX`.