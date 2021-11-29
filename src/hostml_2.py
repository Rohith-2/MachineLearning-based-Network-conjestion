import pickle
import pandas as pd
import os
import math
import random
from pycaret.regression import *
import numpy as np

import context
from src.device import Device, Device_Type
from src.packet import Packet, Packet_Type

class TCPML():
    def __init__(self):
        self.packets_to_send = list()
        self.packets_in_flight = list()
        self.pckts_to_resend = list()
        #self.window_size = random.randint(1,300)
        self.window_size = 1
        self.timeout = 10

        self.ack_recv_flag = False
        self.ack_timeout_flag = False

class HostML(Device):
    def __init__(self, ip:str, buffer_cap=5):
        super().__init__(ip)
        self.connected_router = None
        self.outgoing_buffer = list()
        self.incoming_buffer = list()
        self.buffer_cap = buffer_cap
        self.tcp = TCPML()
        self.def_seg_no = 1

        #model_path = os.path.join('model/model.pickle')   
        self.model = load_model('model/best-model')
        print(self.model)
    
    def link(self,other:Device):
        self.connected_router = other
    
    def get_connected_router(self):
        return self.connected_router
    
    def device_type(self):
        return Device_Type.HOST

    def send_pckt(self,pckt:Packet):
        self.tcp.packets_to_send.append(pckt)
    
    def send_random_packet(self,to_device:Device):
        pckt = Packet(self.def_seg_no,self,to_device,Packet_Type.DATA)
        self.send_pckt(pckt)
        self.def_seg_no = self.def_seg_no + 1

    def receive_pckt(self,pckt:Packet):
        if len(self.incoming_buffer) < self.buffer_cap:
            self.incoming_buffer.append(pckt)
    
    def __str__(self):
        msg = "Host IP: {}\r\n".format(self.ip)
        msg = msg + "Connected to {}\r\n".format(self.connected_router.get_ip())

        return msg
    
    def step(self):
        super().step()

        self.tcp.ack_recv_flag = False     
        self.tcp.ack_timeout_flag = False

        # handle incoming packets
        for pckt in self.incoming_buffer:      # iterating through the packets in incoming buffer
            if pckt.get_pckt_type() == Packet_Type.DATA:  # checking for data 
                # send ack packet
                ack_pack = Packet(pckt.get_seg_no(),pckt.get_to(),pckt.get_from(),Packet_Type.ACK) # instantiating ACK pkt
                self.outgoing_buffer.append(ack_pack)  # appending ACKs into outgoing buffer
                # print("Host {} received packet {} from host {} and sent ACK.".format(self.get_ip(), pckt.get_seg_no(), pckt.get_from().get_ip()))
                pass
            
            elif pckt.get_pckt_type() == Packet_Type.ACK:  # checking if pkt is ACK pkt
                # remove packet from packets in flight and packets to send
                seg_no = pckt.get_seg_no() # storing current ptkt's seg_no
                
                index = -1
                for i in range(len(self.tcp.packets_in_flight)): # iterating through pkts in flight
                    pckt2 = self.tcp.packets_in_flight[i][0]
                    if pckt2.get_seg_no() == seg_no:  # checks for current pkt 
                        index = i             # index = 1 if current pkt is found
                        break
                
                if index >= 0:
                    self.tcp.timeout = self.clock-self.tcp.packets_in_flight[i][1]     # set tcp timeout adaptively
                    self.tcp.packets_in_flight.pop(index) # pop pkt from pkt_in_flight
                
                index = -1
                for i in range(len(self.tcp.packets_to_send)): # iterating through pkts_to_send to find current pkt
                    pckt2 = self.tcp.packets_to_send[i]
                    if pckt2.get_seg_no() == seg_no:
                        index = i    #update index if found
                        break
                
                if index >= 0:
                    self.tcp.packets_to_send.pop(index) # pop it from the pkts_to_send
                
                # print("Host {} received ACK from host {}.".format(self.get_ip(), pckt.get_from().get_ip()))
                self.tcp.ack_recv_flag = True
                pass

        self.incoming_buffer.clear()

        # resend any timed out packets
        for i in range(len(self.tcp.packets_in_flight)): # iterates through remaining pkts in flight
            pckt,t = self.tcp.packets_in_flight[i] 
            if self.clock - t> self.tcp.timeout:   # if time lived exceeds timeout pkt must be resent
                self.tcp.pckts_to_resend.append(i) # append pkt in pkt_to_resend
        
        for i in self.tcp.pckts_to_resend:          # iterates through pkts_to_resend
            pckt = self.tcp.packets_in_flight[i][0]     
            self.tcp.packets_to_send.insert(0,pckt) # inserts pkts to pkts_to_send
            # print("Host {} resending packet {} due to timeout.".format(self.get_ip(),pckt.get_seg_no()))
            pass
        
        for i in sorted(self.tcp.pckts_to_resend,reverse=True):
            del self.tcp.packets_in_flight[i]       # insert pkts sent to pkts_in_flight

        # reset window size and ssthresh in case of timeout
        if len(self.tcp.pckts_to_resend) > 0:  # if length of pkts to be sent>0
            self.tcp.ack_timeout_flag = True   # ACK Timeout flag is set to true
            
        # Using the calculated values for prediction
        # predict new window size using model
        model_input = np.array([[
            self.tcp.window_size,
            self.tcp.ack_recv_flag,
            self.tcp.ack_timeout_flag
        ]])

        model_output = self.model.predict(pd.DataFrame(model_input,columns=['window_size', 'ack_received', 'ack_timeout']))

        self.tcp.window_size = int(model_output[0])

        self.tcp.pckts_to_resend.clear()

        if self.tcp.window_size < 1:
            self.tcp.window_size = 1    # minimum window size

        # send packets
        # send packets only if there are no packets in flight
        if len(self.tcp.packets_in_flight) == 0:

            for i in range(self.tcp.window_size):
                if len(self.tcp.packets_to_send) == 0:
                    break

                pckt = self.tcp.packets_to_send.pop(0)
                self.outgoing_buffer.append(pckt)
                self.tcp.packets_in_flight.append((pckt,self.clock))

            for pckt in self.outgoing_buffer:
                if pckt.get_pckt_type() == Packet_Type.DATA:
                    # print("Host {} sent packet {} to host {}.".format(self.get_ip(), pckt.get_seg_no(), pckt.get_to().get_ip()))
                    pass
                self.connected_router.receive_pckt(pckt)
            
            self.outgoing_buffer.clear()


if __name__ == "__main__":
    h = HostML("1")
    h.step()
