//
// Created by arabica on 09.12.2020.
//
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>

#ifndef SK_COMMUNICATION_H
#define SK_COMMUNICATION_H
void send_msg(char * buff, int cfd){
    int rc = 0;
    while (rc<7){
        rc += write(cfd, buff+rc, 1);
        }


}

void recive_msg(char * buff, int cfd){
    int rc =0;
    int r = 1;
    while(r!=0){
        r = read(cfd, buff+rc, 1);
        rc+=r;
        if(*(buff+rc-1) == '\n'){
            return;
            break;
            }
    }
    return;
}
#endif //SK_COMMUNICATION_H
