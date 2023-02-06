#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:00:25 2022

@author: michelev
"""

import requests

import sys
import ssl

#%%

def getResponseProxies(url, headers, proxies):
    
    # get response
    response = requests.get(url, headers = headers, proxies = proxies)
    
    # response could either be None
    res_is_none = response == None
    res_wrong = True
    if res_is_none == False:
        res_wrong = response.status_code != 200
    while (res_is_none == True) | (res_wrong == True):

        response = requests.get(url, proxies = proxies)
        res_is_none = response == None
        res_wrong = True
        if res_is_none == False:
            res_wrong = response.status_code != 200

    return response

def getOpenerSERP(proxies):
        
    # print('If you get error "ImportError: No module named \'six\'" install six:\n'+\
    #     '$ sudo pip install six');
    # print('To enable your free eval account and get CUSTOMER, YOURZONE and ' + \
    #     'YOURPASS, please contact sales@brightdata.com')
    
    ssl._create_default_https_context = ssl._create_unverified_context
    if sys.version_info[0]==2:
        import six
        from six.moves.urllib import request
        opener = request.build_opener(
            request.ProxyHandler(
                {'http': 'http://brd-customer-hl_33f39684-zone-novserp:zf5trt1u8njy@zproxy.lum-superproxy.io:22225',
                'https': 'http://brd-customer-hl_33f39684-zone-novserp:zf5trt1u8njy@zproxy.lum-superproxy.io:22225'}))
    if sys.version_info[0]==3:
        import urllib.request
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(
                {'http': 'http://brd-customer-hl_33f39684-zone-novserp:zf5trt1u8njy@zproxy.lum-superproxy.io:22225',
                'https': 'http://brd-customer-hl_33f39684-zone-novserp:zf5trt1u8njy@zproxy.lum-superproxy.io:22225'}))

    return opener