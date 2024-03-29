# Husk

A tool for building ~10x smaller, self-hostable, read-only container registries.

## Installing

```
$ pip install husk-cli
```

## Usage

Use Husk to extract a list of images from your local machine:

```
$ husk extract 'your-domain.com/*'
```

This creates a `dist` folder with the following files:

```
$ tree dist
dist
├── app
│   ├── blobs
│   │   └── <list of blobs here>
│   └── manifests
│       └── <image names>
│           └── <image versions>
├── docker-compose.yml
└── nginx.conf
```

Copy (or `rsync`) the `dist` folder to your server and spin-up the NGINX server:

```
$ cd dist
$ docker compose up -d
```

That's it! Now you can pull images from your self-hosted container registry:

```
$ docker run --rm -it your-domain.com/your-image-here
```

## Why?

Husk offers many benefits over the standard [Docker Registry](https://hub.docker.com/_/registry):

* Husk is read-only, giving you secure anonymous access out of the box
* Husk only stores blobs you built locally: blobs available on Docker Hub are deferred to Docker Hub (we call this "blob sharding")
* Husk uses NGINX for fast static file hosting/caching
* Husk supports infinite namespaces for images. For example: `your-domain.com/service/saas-product/backend/api:v1`

In contrast, Husk is probably not for you if:

* You frequently need to `docker push` new images
* You don't care about the size of your container registry
* You don't want anonymous access

## Target Audience

Husk is primarily meant for FOSS companies and orgs that want to host their images on their own domains.

Husk can extract any built image, but it really starts to shine when extracting wrapper images,
Alpine-based images, and other size-optimized images.

## Planned Features

* Add flag to disable blob sharding (removes dependency on Docker Hub)
* Package manager aware blob sharding (requires build-time support or other black magic)
* Optional NGINX-based authorization:
  * Instance wide access
  * Namespace wide access
  * Per-image access
* More spec compliant error messages
* Add optional support for the Docker Registry v2 `/catalog` endpoint
* `deploy` or `sync` command for `rsync`ing `dist` folder to a remote server
