"""
Connection to the StableNet server
"""
import http.client
import io
import ssl
import xml.etree.ElementTree as ET
import base64
import getpass
import re

import pandas as pd
import plotly.io as pio
from data_science_package.utils import Helpers

pio.renderers.default = 'iframe'

class StableNetConnection:
    """
    HTTPS Connection to the StableNet server
    """

    def __init__(self, sn_server_ip, sn_server_port, sn_username, sn_password = None):
        ssl._create_default_https_context = ssl._create_unverified_context
        self.__connection = http.client.HTTPSConnection(sn_server_ip, sn_server_port)

        if sn_password is None:
            sn_password = getpass.getpass('Enter stablenet password: ')
        credential_str = sn_username + ':' + sn_password
        credential_bytes = credential_str.encode('ascii')
        credential_bytes_b64 = base64.b64encode(credential_bytes)
        self.__encoded_credentials = credential_bytes_b64.decode('ascii')

    def get_encoded_credentials(self):
        """
        :return: encoded credentials
        """
        return self.__encoded_credentials

    def get_http_connection(self):
        """
        :return: HTTP connection
        """
        return self.__connection

    def test(self):
        """
        Test if connection to StableNet server is successfull.
        """
        response = RequestSender() \
            .with_connection(self) \
            .send_to_endpoint('/rest/info/version') \
            .to_str()
        version = re.search(r'versiontext="([\d\.a-zA-Z \(\)]*?)"', response)
        if version:
            print('Successfully connected to StableNet® Version ' + version.group(1) + '.')
        else:
            print('Could not connect to the StableNet® Server with the received connection info.')


class RequestSender:
    """
    Connection to the StableNet server to send requests
    """
    def __init__(self):
        self.__connection = None
        self.__payload = ''
        self.__headers = {}
        self.__params = {}

    def with_connection(self, connection : StableNetConnection):
        """
        Build a connection to the StableNet server

        :param connection: an instance of StableNetConnection
        :return:
        """
        self.__connection = connection.get_http_connection()
        self.__headers['Authorization'] = 'Basic ' + connection.get_encoded_credentials()
        return self

    def with_tagfilter(self, tagfilter):
        """
        Send a request with a tagfilter
        :param tagfilter:
        :return:
        """
        self.__headers['Content-Type'] = 'application/xml'
        self.__payload = tagfilter
        return self

    def with_xml_header(self):
        """
        Send a request with an XML header

        :return:
        """
        self.__headers['Content-Type'] = 'application/xml'
        return self

    def with_query_param(self, param_name, param_value):
        """
        Send a request with a query parameter

        :param param_name: name of the query parameter
        :param param_value: value of the query parameter
        :return:
        """
        self.__params[param_name] = param_value
        return self

    def with_start_date(self, start_date):
        """
        Send a request with a start date as query parameter
        """
        return self.with_query_param('start', Helpers.to_unix_time(start_date))

    def with_end_date(self, end_date):
        """
        Send a request with an end date as query parameter
        """
        return self.with_query_param('end', Helpers.to_unix_time(end_date))

    def send_to_endpoint(self, endpoint_str):
        """
        Send a request to the StableNet server

        :return: a StableNetResponse
        """
        if self.__connection is None:
            raise TypeError('No connection, can\'t send request.')

        request_type = 'GET' if self.__payload == '' else 'POST'

        orig_endpoint_str = endpoint_str
        for param_name in self.__params.items():
            endpoint_str += '&' + param_name[0] + '=' + param_name[1]
        endpoint_str = endpoint_str.replace('&', '?', 1)

        try:
            self.__connection.request(request_type, endpoint_str, self.__payload, self.__headers)
            return StableNetResponse(self.__connection.getresponse())
        except http.client.RemoteDisconnected:
            print('Due to a long time between requests, the StableNet Server disconnected. '
                  'Retrying...\n')
            return self.send_to_endpoint(orig_endpoint_str)


class StableNetResponse:
    """
    Response from the StableNet server. Can be converted to an XML tree if necessary.
    """
    def __init__(self, response):
        self.__response_str = response.read().decode('utf-8')

    def to_xml_tree(self):
        """
        :return: Response from StableNet server as an XML tree
        """
        return ET.fromstring(self.__response_str)

    def to_str(self):
        """
        :return: Response from StableNet server as a string
        """
        return self.__response_str

def get_data(connection, measurement_id, start_date, end_date):
    """
    Load the data from the server

    :param connection: a StableNetConnection object containing server IP, port, username, and password
    :param measurement_id: ID of the measurement
    :param start_date: start of the desired time frame (Format: YYYY-MM-DD)
    :param end_date: end of the desired time frame (Format: YYYY-MM-DD)
    :return:
    """
    response_str = RequestSender() \
        .with_connection(connection) \
        .with_start_date(start_date) \
        .with_end_date(end_date) \
        .send_to_endpoint("/rest/measurements/aggregateddata/" + str(measurement_id)) \
        .to_str()

    data_response = pd.read_csv(io.StringIO(response_str), sep=";")
    data_response['Time'] = pd.to_datetime(data_response['Time'], unit='ms')

    print('Requested data:', data_response.shape[1], 'columns and', data_response.shape[0], 'rows.')
    return data_response
