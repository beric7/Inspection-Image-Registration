# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:32:36 2021

@author: Admin
"""

class model():
    def __init__(self):
        self.matching = None
        self.device = None
        self.exemplar_num = None
    def set_matching(self, matching):
        self.matching = matching
    def set_device(self, device):
        self.device = device
    def set_exemplar_num(self,exemplar_num):
        self.exemplar_num = exemplar_num
        
    def get_matching(self):
        return self.matching
    def get_device(self):
        return self.device
    def get_exemplar_number(self):
        return self.exemplar_num
    

