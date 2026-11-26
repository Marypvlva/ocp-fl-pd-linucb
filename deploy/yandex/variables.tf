variable "cloud_id" {
  type = string
}

variable "folder_id" {
  type = string
}

variable "operator_cidr" {
  description = "Your public address in CIDR form, e.g. 203.0.113.7/32 (SSH to the coordinator)."
  type        = string
}

variable "ssh_public_key_path" {
  type    = string
  default = "~/.ssh/id_ed25519.pub"
}

variable "prefix" {
  type    = string
  default = "ocp-stand"
}

variable "image_family" {
  type    = string
  default = "ubuntu-2404-lts"
}

variable "platform_id" {
  description = "standard-v3 = Intel Ice Lake."
  type        = string
  default     = "standard-v3"
}

variable "preemptible" {
  description = "Preemptible VMs are cheaper but may be stopped; the study runner is resumable."
  type        = bool
  default     = false
}

variable "subnets" {
  type = map(string)
  default = {
    "ru-central1-a" = "10.10.1.0/24"
    "ru-central1-b" = "10.10.2.0/24"
    "ru-central1-d" = "10.10.3.0/24"
  }
}

variable "coordinator_zone" {
  type    = string
  default = "ru-central1-a"
}

variable "worker_zones" {
  type    = list(string)
  default = ["ru-central1-a", "ru-central1-b", "ru-central1-d"]
}

variable "backup_zones" {
  description = "Zones of the backup-lane edges e2 and e3 (failure domains other than the primary lane)."
  type        = list(string)
  default     = ["ru-central1-b", "ru-central1-d"]
}

variable "worker_hosts" {
  description = "VMs that host worker roles (roles are assigned round-robin; paired local test sampler supports up to 16 roles)."
  type        = number
  default     = 3
}

variable "worker_cores" {
  type    = number
  default = 2
}

variable "worker_memory" {
  type    = number
  default = 2
}

variable "coordinator_cores" {
  description = "One OCP ceremony is single-threaded; parallel episodes need about one vCPU each."
  type        = number
  default     = 4
}

variable "coordinator_memory" {
  type    = number
  default = 8
}

variable "primary_edge_netem_ms" {
  description = "Extra one-way delay on primary-lane edges (0 = the real network only)."
  type        = number
  default     = 0
}

variable "backup_edge_netem_ms" {
  description = "Extra one-way delay on backup-lane edges, e.g. 20 for a regional backup."
  type        = number
  default     = 0
}
