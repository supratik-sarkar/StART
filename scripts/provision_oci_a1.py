#!/usr/bin/env python3
"""Provision OCI Always Free A1 Flex Instance in ap-mumbai-1 for StART v4.5.2.

Zero-cost verified:
- Shape: VM.Standard.A1.Flex (2 OCPU / 12 GB RAM)
- Storage: 50 GB Boot Volume (within 200 GB Always Free limit)
- VCN + Public Subnet + Internet Gateway + Security List (Always Free)
"""

import json
import os
import sys
import time

import oci

CONFIG_PATH = os.path.expanduser("~/.oci/config")
PROFILE = "START_V452_DEPLOY"
SSH_PUBKEY_PATH = os.path.expanduser("~/.ssh/id_ed25519_start_oci.pub")

def main():
    print("=== PROVISIONING OCI ALWAYS FREE A1 COMPUTE FOR StART v4.5.2 ===")
    
    config = oci.config.from_file(CONFIG_PATH, PROFILE)
    token_file = config["security_token_file"]
    with open(token_file) as f:
        token = f.read()
    private_key = oci.signer.load_private_key_from_file(config["key_file"])
    signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
    
    tenancy_id = config["tenancy"]
    compute_client = oci.core.ComputeClient(config, signer=signer)
    network_client = oci.core.VirtualNetworkClient(config, signer=signer)
    
    with open(SSH_PUBKEY_PATH) as f:
        ssh_public_key = f.read().strip()
    
    ad_name = "IrkY:AP-MUMBAI-1-AD-1"
    print(f"Target Availability Domain: {ad_name}")
    
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent

    # 1. Look for existing or create VCN
    vcns = network_client.list_vcns(tenancy_id).data
    target_vcn = None
    for v in vcns:
        if v.display_name == "start-v452-vcn" and v.lifecycle_state in ("AVAILABLE", "PROVISIONING"):
            target_vcn = v
            break
    
    vcn_cidr = f"{10}.{0}.{0}.{0}/16"
    subnet_cidr = f"{10}.{0}.{1}.{0}/24"

    if not target_vcn:
        print(f"Creating VCN: start-v452-vcn ({vcn_cidr})...")
        vcn_details = oci.core.models.CreateVcnDetails(
            cidr_block=vcn_cidr,
            display_name="start-v452-vcn",
            compartment_id=tenancy_id,
            dns_label="startvcn"
        )
        target_vcn = network_client.create_vcn(vcn_details).data
        print(f"VCN created: {target_vcn.id}")
        time.sleep(3)
    else:
        print(f"Using existing VCN: {target_vcn.id}")

    # 2. Look for existing or create Internet Gateway
    igws = network_client.list_internet_gateways(tenancy_id, vcn_id=target_vcn.id).data
    target_igw = None
    for ig in igws:
        if ig.display_name == "start-v452-igw" and ig.lifecycle_state == "AVAILABLE":
            target_igw = ig
            break
            
    if not target_igw:
        print("Creating Internet Gateway: start-v452-igw...")
        igw_details = oci.core.models.CreateInternetGatewayDetails(
            compartment_id=tenancy_id,
            is_enabled=True,
            vcn_id=target_vcn.id,
            display_name="start-v452-igw"
        )
        target_igw = network_client.create_internet_gateway(igw_details).data
        print(f"Internet Gateway created: {target_igw.id}")
        time.sleep(2)
    else:
        print(f"Using existing IGW: {target_igw.id}")

    # 3. Route Table with 0.0.0.0/0 -> IGW
    route_tables = network_client.list_route_tables(tenancy_id, vcn_id=target_vcn.id).data
    default_rt = route_tables[0]
    route_rules = [
        oci.core.models.RouteRule(
            destination="0.0.0.0/0",
            destination_type="CIDR_BLOCK",
            network_entity_id=target_igw.id
        )
    ]
    update_rt_details = oci.core.models.UpdateRouteTableDetails(route_rules=route_rules)
    network_client.update_route_table(default_rt.id, update_rt_details)
    print(f"Updated Route Table: {default_rt.id} with default gateway rule.")

    # 4. Security List with SSH (22), HTTP (80), HTTPS (443), and App (8000)
    sec_lists = network_client.list_security_lists(tenancy_id, vcn_id=target_vcn.id).data
    default_sl = sec_lists[0]
    
    ingress_rules = [
        oci.core.models.IngressSecurityRule(
            protocol="6", # TCP
            source="0.0.0.0/0",
            tcp_options=oci.core.models.TcpOptions(destination_port_range=oci.core.models.PortRange(min=22, max=22)),
            description="SSH"
        ),
        oci.core.models.IngressSecurityRule(
            protocol="6",
            source="0.0.0.0/0",
            tcp_options=oci.core.models.TcpOptions(destination_port_range=oci.core.models.PortRange(min=80, max=80)),
            description="HTTP / Let's Encrypt ACME"
        ),
        oci.core.models.IngressSecurityRule(
            protocol="6",
            source="0.0.0.0/0",
            tcp_options=oci.core.models.TcpOptions(destination_port_range=oci.core.models.PortRange(min=443, max=443)),
            description="HTTPS / TLS"
        ),
        oci.core.models.IngressSecurityRule(
            protocol="6",
            source="0.0.0.0/0",
            tcp_options=oci.core.models.TcpOptions(destination_port_range=oci.core.models.PortRange(min=8000, max=8000)),
            description="StART Web Service"
        ),
    ]
    egress_rules = [
        oci.core.models.EgressSecurityRule(
            protocol="all",
            destination="0.0.0.0/0",
            description="Allow all outbound"
        )
    ]
    update_sl_details = oci.core.models.UpdateSecurityListDetails(
        ingress_security_rules=ingress_rules,
        egress_security_rules=egress_rules
    )
    network_client.update_security_list(default_sl.id, update_sl_details)
    print(f"Updated Security List: {default_sl.id} with ingress ports (22, 80, 443, 8000).")

    # 5. Public Subnet
    subnets = network_client.list_subnets(tenancy_id, vcn_id=target_vcn.id).data
    target_subnet = None
    for s in subnets:
        if s.display_name == "start-v452-public-subnet" and s.lifecycle_state == "AVAILABLE":
            target_subnet = s
            break
            
    if not target_subnet:
        print(f"Creating Public Subnet: start-v452-public-subnet ({subnet_cidr})...")
        subnet_details = oci.core.models.CreateSubnetDetails(
            compartment_id=tenancy_id,
            vcn_id=target_vcn.id,
            cidr_block=subnet_cidr,
            display_name="start-v452-public-subnet",
            route_table_id=default_rt.id,
            security_list_ids=[default_sl.id],
            prohibit_public_ip_on_vnic=False,
            dns_label="startsub"
        )
        target_subnet = network_client.create_subnet(subnet_details).data
        print(f"Subnet created: {target_subnet.id}")
        time.sleep(3)
    else:
        print(f"Using existing Subnet: {target_subnet.id}")

    # 6. Find Ubuntu 24.04 ARM64 image
    images = compute_client.list_images(tenancy_id, shape="VM.Standard.A1.Flex").data
    target_image = None
    for img in images:
        if "Canonical-Ubuntu-24.04-aarch64" in img.display_name and "Minimal" not in img.display_name:
            target_image = img
            break
    if not target_image:
        for img in images:
            if "Ubuntu-24.04" in img.display_name or "Oracle-Linux-9" in img.display_name:
                target_image = img
                break
    print(f"Selected OS Image: {target_image.display_name} ({target_image.id})")

    # 7. Check for existing compute instance
    instances = compute_client.list_instances(tenancy_id).data
    target_instance = None
    for inst in instances:
        if inst.display_name == "start-v452-backend" and inst.lifecycle_state not in ("TERMINATED", "TERMINATING"):
            target_instance = inst
            break

    if not target_instance:
        print("Launching Compute Instance: start-v452-backend (VM.Standard.A1.Flex, 2 OCPU, 12 GB RAM, 50 GB Boot Volume)...")
        shape_config = oci.core.models.LaunchInstanceShapeConfigDetails(
            ocpus=2.0,
            memory_in_gbs=12.0
        )
        source_details = oci.core.models.InstanceSourceViaImageDetails(
            image_id=target_image.id,
            boot_volume_size_in_gbs=50
        )
        vnic_details = oci.core.models.CreateVnicDetails(
            subnet_id=target_subnet.id,
            assign_public_ip=True,
            display_name="start-v452-vnic"
        )
        launch_details = oci.core.models.LaunchInstanceDetails(
            compartment_id=tenancy_id,
            availability_domain=ad_name,
            display_name="start-v452-backend",
            shape="VM.Standard.A1.Flex",
            shape_config=shape_config,
            source_details=source_details,
            create_vnic_details=vnic_details,
            metadata={"ssh_authorized_keys": ssh_public_key}
        )
        
        try:
            target_instance = compute_client.launch_instance(launch_details).data
            print(f"Instance launched: {target_instance.id}, State: {target_instance.lifecycle_state}")
        except oci.exceptions.ServiceError as e:
            if "Out of host capacity" in str(e) or "Capacity" in str(e) or e.status == 500:
                print("PUBLIC DEPLOYMENT BLOCKED — OCI ALWAYS FREE CAPACITY UNAVAILABLE")
                sys.exit(2)
            else:
                raise e
    else:
        print(f"Using existing Instance: {target_instance.id}, State: {target_instance.lifecycle_state}")

    # 8. Wait for instance to become RUNNING
    print("Waiting for instance to reach RUNNING state...")
    while True:
        target_instance = compute_client.get_instance(target_instance.id).data
        print(f"Instance status: {target_instance.lifecycle_state}")
        if target_instance.lifecycle_state == "RUNNING":
            break
        elif target_instance.lifecycle_state in ("TERMINATED", "TERMINATING"):
            print("PUBLIC DEPLOYMENT BLOCKED — OCI INSTANCE TERMINATED UNEXPECTEDLY")
            sys.exit(1)
        time.sleep(10)

    # 9. Get Public IP
    vnic_attachments = compute_client.list_vnic_attachments(tenancy_id, instance_id=target_instance.id).data
    vnic_id = vnic_attachments[0].vnic_id
    vnic = network_client.get_vnic(vnic_id).data
    public_ip = vnic.public_ip
    print(f"SUCCESS: Instance is RUNNING with Public IP: {public_ip}")

    # Save details
    deployment_info = {
        "instance_id": target_instance.id,
        "instance_name": target_instance.display_name,
        "shape": target_instance.shape,
        "ocpus": 2,
        "memory_gbs": 12,
        "public_ip": public_ip,
        "availability_domain": ad_name,
        "region": "ap-mumbai-1",
        "tenancy": "supratiksarkar",
        "ssh_user": "ubuntu" if "ubuntu" in target_image.display_name.lower() else "opc",
        "sslip_domain": f"{public_ip}.sslip.io"
    }
    
    info_path = ROOT / "start_output" / "v452_remote_release" / "oracle_instance_info.json"
    with open(info_path, "w") as f:
        json.dump(deployment_info, f, indent=2)
    print(f"Instance deployment metadata written to {info_path}")

if __name__ == "__main__":
    main()
