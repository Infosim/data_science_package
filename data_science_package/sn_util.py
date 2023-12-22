from typing import Union, List, Tuple
import copy
from IPython.display import display
import requests
from requests.auth import HTTPBasicAuth
from getpass import getpass
from xml.etree import ElementTree
from PIL import Image
import warnings
import pandas as pd
import datetime


class StablenetClient:
    def __init__(self, *, server_ip, server_port, username, password=None) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.username = username
        self.__password = password if password else getpass("Enter stablenet password: ")
        warnings.filterwarnings("ignore")

    def get_geocoordinates(self, tag):
        url = f"https://{self.server_ip}:{self.server_port}/rest/tag/query"
        headers = {'Content-Type': 'application/xml'}
        resp = requests.post(url, data=tag, headers=headers, verify=False, auth=HTTPBasicAuth(self.username, self.__password))
        resp: ElementTree = ElementTree.fromstring(resp.content)
        return resp

    def get_geocoordinates_for_graph(self, graph, id_map):
        tag = get_geocoordinatestag(graph.nodes.data())
        url = f"https://{self.server_ip}:{self.server_port}/rest/tag/query"
        headers = {'Content-Type': 'application/xml'}
        resp = requests.post(url, data=tag, headers=headers, verify=False, auth=HTTPBasicAuth(self.username, self.__password))
        resp: ElementTree = ElementTree.fromstring(resp.content)
        for entry in resp.iter('taggableoutputentry'):
            obid = entry.find("reference").attrib["obid"]
            lat = entry.find("tags/tag[@key='Device Geo Latitude']")
            long = entry.find("tags/tag[@key='Device Geo Longitude']")
            if not lat is None:
                lat = lat.attrib['value']
            if not long is None:
                long = long.attrib['value']
            if not lat is None and not long is None:
                graph.nodes[id_map[obid]]['lat'] = float(lat)
                graph.nodes[id_map[obid]]['long'] = float(long)
            else:
                graph.remove_node(id_map[obid])
                id_map.pop(obid, None)
        return graph, id_map

    def list_weathermaps(self) -> ElementTree:
        resp = requests.get("https://" + self.server_ip + ":" + self.server_port + "/rest/weathermaps/list",
                            verify=False, auth=HTTPBasicAuth(self.username, self.__password))
        return ElementTree.fromstring(resp.content)


    def get_weathermap(self, wmap_id: Union[str, int]) -> ElementTree:
        url = "https://{}:{}/rest/weathermaps/get/{}".format(self.server_ip, self.server_port, wmap_id)
        resp = requests.get(url, verify=False, auth=HTTPBasicAuth(self.username, self.__password))
        wmap: ElementTree = ElementTree.fromstring(resp.content)

        if wmap.tag == "error":
            raise AttributeError("Error: WeatherMap with id {} does not exist on server {}:{}."
                                 .format(wmap_id, self.server_ip, self.server_port))
        elif wmap.tag == "html":
            raise AttributeError("Error: Wrong credentials used.")
        return wmap

    def get_neighborhoodmap(self, device_ids: List[int]) -> ElementTree:
        devicelist = (','.join(str(x) for x in device_ids))

        url = "https://{}:{}/rest/devices/neighborhood/weathermap/{}".format(self.server_ip, self.server_port, devicelist)
        resp = requests.get(url, verify=False, auth=HTTPBasicAuth(self.username, self.__password))
        wmap: ElementTree = ElementTree.fromstring(resp.content)

        if wmap.tag == "error":
            raise AttributeError("Error: NeighborhoodMap with ids {} does not exist on server {}:{}."
                                 .format(devicelist, self.server_ip, self.server_port))
        elif wmap.tag == "html":
            raise AttributeError("Error: Wrong credentials used.")
        return wmap


    def add_weathermap(self, wmap: ElementTree) -> ElementTree:
        add_pws_flag(wmap)
        finalMap = ElementTree.tostring(wmap).decode('utf-8')
        resp = requests.post("https://" + self.server_ip + ":" + self.server_port + "/rest/weathermaps/add/",
                             verify=False, auth=HTTPBasicAuth(self.username, self.__password),
                             data=finalMap, headers={'Content-Type': 'application/xml'})
        return ElementTree.fromstring(resp.content)


    def get_wmap_png(self, obid: Union[str, int]):
        """Usage: display(get_wmap_png(obid))"""
        response = requests.get("https://" + self.server_ip + ":" + self.server_port + "/rest/weathermaps/plot/" + obid,
                                verify=False, auth=HTTPBasicAuth(self.username, self.__password),
                                headers={'Content-Type': 'application/xml'}, stream=True)
        response.raw.decode_content = True
        return Image.open(response.raw)

    def upload_weathermap(self, wmap: ElementTree, map_name: str = None, node_colors: Tuple[str, dict] = None, edge_colors: Tuple[str, dict] = None, use_positions=True, update_icons: bool = True, node_ranges: List[float] =None, link_ranges: List[float]=None):
        wmap_object = copy.deepcopy(wmap)
        if map_name is not None:
            wmap_object.set('name', map_name)
        if update_icons:
            for node in wmap_object.findall('weathermapnodes/weathermapnode'):
                node.set('icon', '114')
        if use_positions:
            wmap_object.set('layoutmode', 'custom')
        else:
            wmap_object.set('layoutmode', 'organic')
        color_links(wmap_object, edge_colors, link_ranges=range_string(link_ranges))  # Appends stat-tags to edges
        color_nodes(wmap_object, node_colors, node_ranges=range_string(node_ranges))  # Appends stat-tags to nodes
        uploaded_wmap = self.add_weathermap(wmap_object)  # Adds modified weathermap via POST-Request
        print(f"New weathermap has obid {uploaded_wmap.get('obid')}")
        return uploaded_wmap

    def plot_wmap(self, uploaded_wmap: ElementTree):
        im = self.get_wmap_png(obid=uploaded_wmap.get("obid"))  # Requests a png-plot of the newly added weathermap
        display(im)

    def load_logdata(self, num_logs=10000, starttime='2022-07-06 00:00:00',endtime='2022-07-20 00:00:00'):
        start_utc = str(int(1000000*datetime.datetime.strptime(starttime, "%Y-%m-%d %H:%M:%S").timestamp()))
        end_utc = str(int(1000000*datetime.datetime.strptime(endtime, "%Y-%m-%d %H:%M:%S").timestamp()))

        warnings.filterwarnings("ignore")
        response = requests.get(f"https://{self.server_ip}:{self.server_port}/rest/events/syslogs", params={"length": num_logs,
                                                                                                            "start": start_utc,
                                                                                                            "end": end_utc},
                                verify=False, auth=HTTPBasicAuth(self.username, self.__password))

        log_df = pd.read_xml(response.content)
        cols = ['deviceid', 'message', 'syslogpriority', 'syslogtime']
        log_df = log_df[cols]
        datetimeformat = '%Y %b %d %H:%M:%S'

        mask = log_df["syslogtime"].str.contains("w")
        log_df = log_df.drop(log_df[mask].index)
        log_df.reset_index(drop=True, inplace=True)

        # impute datetime from message if the datetime is contained in the message but not available in the dateime col
        pattern = "(?<=\w):(?=\D)"
        mask = log_df["syslogtime"].str.endswith(":")
        log_df.loc[mask, "syslogtime"] = log_df.loc[mask]["message"].str.split(pattern).str[1].str.split(".").str[0]
        log_df.loc[mask, "message"] = log_df.loc[mask]["message"].str.split(pattern).str[2:].str.join(":")
        log_df["syslogtime"] = "2022 " + log_df["syslogtime"]

        # changing the syslogtime from str to datetime objects for later plots
        # drop all columns where syslogtime doesn't match  datetimeformat
        log_df = log_df[log_df['syslogtime'].astype(str).str.match(r'\d{4} \w+\s+\d+ \d\d:\d\d:\d\d')]
        log_df["syslogtime"] = pd.to_datetime(log_df["syslogtime"], format=datetimeformat)

        return log_df

    def get_device_names_and_tags(self, log_df):
        data = []
        for device in log_df["deviceid"].unique():
            response = requests.get(f"https://{self.server_ip}:{self.server_port}/rest/devices/get/{device}",
                                    verify=False, auth=HTTPBasicAuth(self.username, self.__password))
            root = ElementTree.fromstring(response.content)
            name = root.attrib["name"]
            alltags = root.find("tags").findall("tag")
            row = []
            for tag in alltags:
                row.append(tag.attrib["value"])
            data.append([device, name, row])
        return pd.DataFrame(data, columns=["deviceid", "devicename", "tags"])


def range_string(ranges: List[float]):
    if ranges is None:
        return None
    # 0.60/1.00/5000,0.40/0.60/4000,0.20/0.40/500,0.0/0.20
    if len(ranges) != 5:
        # TODO make general
        raise NotImplementedError("Ranges should be of length 5")
    steps = ['', '500', '4000', '5000']
    out = ""
    for (start, end), step in zip(zip(ranges, ranges[1:]), steps):
        if start >= end or start < 0.0 or end > 1.0:
            raise ValueError("Range steps should be ascending and within [0.0, 1.0]")
        out = f"{start:.2f}/{end:.2f}{f'/{step}' if len(step) > 0 else ''}" + (f",{out}" if len(out) > 0 else "")
    return out


# static
def add_pws_flag(wmap: ElementTree):
    """For identifying weathermaps created by this script - sets Level-0-Tag to given value"""
    level_0_val = "Topology Analysis Demo"
    tags = wmap.find("tags")
    if tags is None:
        tags = ElementTree.SubElement(wmap, 'tags')
    level_0_tag = tags.find("tag[@key='Weather Map Level 0']")
    if level_0_tag is not None:
        tags.remove(level_0_tag)
    ElementTree.SubElement(tags, 'tag', {'id': '2300', 'key': 'Weather Map Level 0', 'value': level_0_val})


# static
def append_stat_tag(wmapobject, title, ranges, value, defaultstate='0'):
    stat_attrs = {
        'metrickey': 'AVG_LOST_PERC',
        'type': 'measurementstat',
        'title': title,
        'ranges': ranges,
        'defaultstate': defaultstate, # blue = 2000
        'showaslabel': 'false'
    }
    statistic = ElementTree.SubElement(wmapobject.find('statistics'), 'statistic', stat_attrs)
    ElementTree.SubElement(statistic, 'reference', {'obid': '13475', 'domain': 'measurement'})
    ElementTree.SubElement(statistic, 'metricscale', {'add': value, 'multiply': '0'})


# static
def color_links(wmap_object: ElementTree, used_metric: Tuple[str, dict], link_ranges=None):
    ranges = '0.60/1.00/5000,0.40/0.60/4000,0.20/0.40/500,0.0/0.20' if link_ranges is None else link_ranges
    if used_metric is None:
        return
    metric_name, metric_dict = used_metric
    for link in wmap_object.findall('weathermaplinks/weathermaplink'):
        if metric_dict.get((link.get('sourcenode'), link.get('destinationnode'))) is not None:
            link_value = str(metric_dict.get((link.get('sourcenode'), link.get('destinationnode'))))
        else:
            link_value = str(metric_dict.get((link.get('destinationnode'), link.get('sourcenode'))))

        append_stat_tag(wmapobject=link, title=metric_name, ranges=ranges, value=link_value)


# static
def color_nodes(wmap_object: ElementTree, used_metric: Tuple[str, dict], node_ranges=None, defaultstate='0'):
    ranges = '0.60/1.00/5000,0.40/0.60/4000,0.20/0.40/500,0.0/0.20' if node_ranges is None else node_ranges
    if used_metric is None:
        return
    metric_name, metric_dict = used_metric
    for node in wmap_object.findall('weathermapnodes/weathermapnode'):
        append_stat_tag(wmapobject=node, title=metric_name, ranges=ranges, value=str(metric_dict.get(node.get('obid'))), defaultstate=defaultstate)

def get_valuetagfilter(key, value):
    return  f"<valuetagfilter filtervalue=\"{value}\"> <tagcategory key=\"{key}\"/> </valuetagfilter>"

def get_geocoordinatestag(node_data):
    tag = f"""<taggablelistqueryinput domain="Device">
    <tagcategories>
		<tagcategory id="71" key="Device Geo Latitude"/>
		<tagcategory id="72" key="Device Geo Longitude"/>
	</tagcategories>
	<ortagfilter>
    {" ".join([get_valuetagfilter("Device ID", obid['device_id']) for _, obid in node_data])}
	</ortagfilter>
    </taggablelistqueryinput>"""
    return tag
