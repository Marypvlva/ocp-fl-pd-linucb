# Distributed stand for the OCP-FL + AF-PD-LinUCB live study in Yandex Cloud.
#
# One coordinator (OCP ceremony in a local EVM, planner, verifier, launcher),
# worker VMs spread over three zones, two primary-lane edges in the
# coordinator's zone and two backup-lane edges in other zones. Only the
# coordinator has a public address; the others reach the Internet through a NAT
# gateway (packages, NTP) and are reached by SSH from the coordinator.

terraform {
  required_version = ">= 1.5"
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.120"
    }
  }
}

provider "yandex" {
  # Authentication: export YC_TOKEN=$(yc iam create-token) or set
  # service_account_key_file; cloud_id/folder_id from variables.
  cloud_id  = var.cloud_id
  folder_id = var.folder_id
  zone      = var.coordinator_zone
}

data "yandex_compute_image" "ubuntu" {
  family = var.image_family
}

resource "yandex_vpc_network" "stand" {
  name = "${var.prefix}-net"
}

resource "yandex_vpc_gateway" "nat" {
  name = "${var.prefix}-nat"
  shared_egress_gateway {}
}

resource "yandex_vpc_route_table" "egress" {
  name       = "${var.prefix}-egress"
  network_id = yandex_vpc_network.stand.id
  static_route {
    destination_prefix = "0.0.0.0/0"
    gateway_id         = yandex_vpc_gateway.nat.id
  }
}

resource "yandex_vpc_subnet" "zone" {
  for_each       = var.subnets
  name           = "${var.prefix}-${each.key}"
  zone           = each.key
  network_id     = yandex_vpc_network.stand.id
  v4_cidr_blocks = [each.value]
  route_table_id = yandex_vpc_route_table.egress.id
}

resource "yandex_vpc_security_group" "stand" {
  name       = "${var.prefix}-sg"
  network_id = yandex_vpc_network.stand.id

  ingress {
    description    = "SSH to the coordinator from the operator"
    protocol       = "TCP"
    port           = 22
    v4_cidr_blocks = [var.operator_cidr]
  }
  ingress {
    description       = "All traffic between stand VMs (roles listen on ephemeral ports)"
    protocol          = "ANY"
    predefined_target = "self_security_group"
    from_port         = 0
    to_port           = 65535
  }
  egress {
    description    = "Packages, NTP, OS updates through NAT"
    protocol       = "ANY"
    v4_cidr_blocks = ["0.0.0.0/0"]
    from_port      = 0
    to_port        = 65535
  }
}

locals {
  workers = {
    for i in range(var.worker_hosts) :
    "worker-${i}" => { zone = var.worker_zones[i % length(var.worker_zones)], cores = var.worker_cores, memory = var.worker_memory, nat = false, netem_ms = 0, kind = "worker" }
  }
  edges = {
    "edge-0" = { zone = var.coordinator_zone, cores = 2, memory = 2, nat = false, netem_ms = var.primary_edge_netem_ms, kind = "edge" }
    "edge-1" = { zone = var.coordinator_zone, cores = 2, memory = 2, nat = false, netem_ms = var.primary_edge_netem_ms, kind = "edge" }
    "edge-2" = { zone = var.backup_zones[0], cores = 2, memory = 2, nat = false, netem_ms = var.backup_edge_netem_ms, kind = "edge" }
    "edge-3" = { zone = var.backup_zones[1 % length(var.backup_zones)], cores = 2, memory = 2, nat = false, netem_ms = var.backup_edge_netem_ms, kind = "edge" }
  }
  nodes = merge(
    { "coordinator" = { zone = var.coordinator_zone, cores = var.coordinator_cores, memory = var.coordinator_memory, nat = true, netem_ms = 0, kind = "coordinator" } },
    local.workers,
    local.edges,
  )
}

resource "yandex_compute_instance" "node" {
  for_each    = local.nodes
  name        = "${var.prefix}-${each.key}"
  hostname    = "${var.prefix}-${each.key}"
  zone        = each.value.zone
  platform_id = var.platform_id
  labels      = { stand = var.prefix, kind = each.value.kind }

  resources {
    cores         = each.value.cores
    memory        = each.value.memory
    core_fraction = 100
  }

  scheduling_policy {
    preemptible = var.preemptible
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.ubuntu.id
      size     = 20
      type     = "network-ssd"
    }
  }

  network_interface {
    subnet_id          = yandex_vpc_subnet.zone[each.value.zone].id
    nat                = each.value.nat
    security_group_ids = [yandex_vpc_security_group.stand.id]
  }

  metadata = {
    user-data = templatefile("${path.module}/cloud-init.yaml", {
      ssh_key         = trimspace(file(pathexpand(var.ssh_public_key_path)))
      netem_ms        = each.value.netem_ms
      netem_jitter_ms = ceil(each.value.netem_ms / 5)
    })
  }
}

output "coordinator_public_ip" {
  value = yandex_compute_instance.node["coordinator"].network_interface[0].nat_ip_address
}

output "inventory" {
  value = {
    coordinator = yandex_compute_instance.node["coordinator"].network_interface[0].ip_address
    workers     = [for i in range(var.worker_hosts) : yandex_compute_instance.node["worker-${i}"].network_interface[0].ip_address]
    edges       = [for i in range(4) : yandex_compute_instance.node["edge-${i}"].network_interface[0].ip_address]
    zones       = { for k, v in local.nodes : k => v.zone }
  }
}
