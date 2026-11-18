import xml.etree.ElementTree as ET
import os

URDF_FILE = "/home/lp531/isaac_assets/z1/z1.urdf"  # 改成你 URDF 路径

if not os.path.exists(URDF_FILE):
    raise FileNotFoundError(f"URDF 文件不存在: {URDF_FILE}")

tree = ET.parse(URDF_FILE)
root = tree.getroot()

# 解析所有 link
links = set()
for link in root.findall('link'):
    name = link.attrib.get('name')
    links.add(name)

print(f"Found {len(links)} links:")
for name in sorted(links):
    print(f" - {name}")

# 解析所有 joint
joints = []
for joint in root.findall('joint'):
    j_name = joint.attrib.get('name')
    j_type = joint.attrib.get('type')
    parent = joint.find('parent').attrib.get('link') if joint.find('parent') is not None else None
    child = joint.find('child').attrib.get('link') if joint.find('child') is not None else None
    joints.append({
        'name': j_name,
        'type': j_type,
        'parent': parent,
        'child': child
    })

print(f"\nFound {len(joints)} joints:")
for j in joints:
    parent_exists = j['parent'] in links
    child_exists = j['child'] in links
    print(f" - {j['name']} ({j['type']}): parent={j['parent']} [{'OK' if parent_exists else 'MISSING'}], child={j['child']} [{'OK' if child_exists else 'MISSING'}]")

# 检查重复 joint
joint_names = [j['name'] for j in joints]
duplicates = set([name for name in joint_names if joint_names.count(name) > 1])
if duplicates:
    print(f"\n⚠️ Duplicate joint names found: {', '.join(duplicates)}")
else:
    print("\nNo duplicate joint names found.")

# 检查是否有 joint 的 parent 或 child 缺失
missing_links = [j for j in joints if j['parent'] not in links or j['child'] not in links]
if missing_links:
    print("\n⚠️ Some joints reference missing links:")
    for j in missing_links:
        print(f" - {j['name']}: parent={j['parent']}, child={j['child']}")
else:
    print("\nAll joint parent/child links are valid.")
