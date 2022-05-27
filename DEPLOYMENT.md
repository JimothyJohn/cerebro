# Cloudflare Tunnel Deployment

## Linux

You'll need an [existing Cloudflare account and domain](https://support.cloudflare.com/hc/en-us/articles/201720164-Creating-a-Cloudflare-account-and-adding-a-website). Complete instructions can be found [here](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup).

```bash
# Download cloudflared for Linux amd64
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb # Install cloudflared
rm cloudflared-linux-amd64.deb # Remove installer file
cloudflared update # update binaries
cloudflared login # obtain .pem key
cloudflared tunnel create yolo # create tunnel id
```

After the last step you should see an output that ends with something like:

"Created tunnel yolo with id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

Put the id string into [config.yml](config.yml):

```yml
url: http://localhost:5000
tunnel: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
credentials-file: /home/<your-username>/.cloudflared/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.json
```

And copy the file to your cloudflared directory:

```bash
cp config.yml ~/.cloudflared
```

Set up the DNS with your pre-existing domain and run!

```bash
cloudflared tunnel route dns yolo yolo.your.domain
cloudflared tunnel run yolo
# Test with a picture
curl https://yolo.your.domain/predict -X POST -F input=@docs/zidane.jpg
```

Set up as a service for persistent activation

```bash
cloudflared service install
systemctl enable cloudflared
systemctl start cloudflared
```
