//
// Created by arabica on 16.12.2020.
//

#ifndef SK_SERVER_H
#define SK_SERVER_H

#include <iostream>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>
#include <ctype.h>
#include <vector>
class server {
public:
    fd_set		fds;
    fd_set  fds_w;
    int begin_server(int port);

    static int accept_new_user(std::vector<int> *users, int g_socket_fd);

    static void update_fdset(fd_set *fds, int *fd_max, std::vector<int> users, int g_socket_fd);

    static int create_s_socket(struct sockaddr_in *sock, int port);

    static int accept_con(int socket_fd, struct sockaddr_in *r_addr);

    static int handle_clients(fd_set *fds, std::vector<int> users);
    static void send_msg(char * buff, int cfd);
    static void recive_msg(char * buff, int cfd);

};


#endif //SK_SERVER_H
