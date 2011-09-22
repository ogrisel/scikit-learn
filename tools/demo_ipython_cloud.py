import os
import sys

from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.compute.deployment import MultiStepDeployment, ScriptDeployment, SSHKeyDeployment

provider_name = sys.argv[1].upper()

provider = getattr(Provider, provider_name)

client_id = os.environ[provider_name + '_ID']
client_secret = os.environ[provider_name + '_SECRET']

conn = get_driver(provider)(client_id, client_secret)

# a task that first installs the ssh key, and then runs the script
msd = MultiStepDeployment([
    # This key will be added to the authorized keys for the root user
    # (/root/.ssh/authorized_keys)
    SSHKeyDeployment(open(os.path.expanduser("~/.ssh/id_rsa.pub")).read()),
    ScriptDeployment("apt-get -y install numpy scipy"),
])

images = conn.list_images()
sizes = conn.list_sizes()

# deploy_node takes the same base keyword arguments as create_node.
# TODO: introspect the list of images to find a recent ubuntu version
#node = conn.deploy_node(name='test', image=images[0], size=sizes[0], deploy=msd)

