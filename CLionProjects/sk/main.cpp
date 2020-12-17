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
#include "communication.h"
#include "server.h"

int			handle_clients( fd_set *fds, std::vector<int> users)
{

    std::cout<<users.size()<<std::endl;
    for (int i = 0 ; i< users.size(); i++)
    {

        char buff[1024];
        if (FD_ISSET(users[i], fds))
        {
            recive_msg(buff, users[i]);
            std::cout<<buff<<std::endl;
            close(users[i]);
        }
    }
    return (0);
}

int			accept_con(int socket_fd, struct sockaddr_in *r_addr)
{
    socklen_t		addrlen;
    int			out_fd;

    addrlen = sizeof(struct sockaddr_in);
    if ((out_fd = accept(socket_fd,
                         (struct sockaddr *)r_addr,
                         (socklen_t*)&addrlen)) == -1)
        return (-1);
    return (out_fd);
}

int	create_s_socket(struct sockaddr_in *sock, int port)
{
    int	socket_fd;
    int	en;

    memset(sock, 0, sizeof(*sock));
    if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
        return (-1);
    en = 1;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &en, sizeof(int)) < 0)
        return (-1);
    sock->sin_family = AF_INET;
    sock->sin_addr.s_addr = INADDR_ANY;
    sock->sin_port = htons(port);
    if (bind(socket_fd, (struct sockaddr *)sock, sizeof(*sock)) < 0)
        return (-1);
    return (socket_fd);
}

static void	update_fdset(fd_set *fds, int *fd_max, std::vector<int> users, int g_socket_fd)
{


    *fd_max = g_socket_fd;
    FD_ZERO(fds);
    FD_SET(g_socket_fd, fds);

    for (int i = 0 ; i< users.size(); i++)
    {
        FD_SET(users[i], fds);
        *fd_max = (users[i] > *fd_max ? users[i]: *fd_max);

    }
}

int	accept_new_user(std::vector<int> * users, int g_socket_fd)
{
    struct sockaddr_in	r_addr;
    socklen_t		addrlen;
    int			nsock;

    nsock = accept_con(g_socket_fd, &r_addr);
    (*users).push_back(nsock);
    std::cout<< "Incoming connection from %s"<<inet_ntoa(r_addr.sin_addr)<<std::endl;
}

int begin_server(int port){
    fd_set		fds;
    int			fd_max, g_socket_fd;
    std::vector<int> users;
    struct sockaddr_in	l_addr;

    if ((g_socket_fd = create_s_socket(&l_addr, port)) == -1)
        return (-1);
    listen(g_socket_fd, 10);
    //init_handler(&h, &users, &chans);
    while (1)
    {

        update_fdset(&fds, &fd_max, users, g_socket_fd);
        if (select(fd_max + 1, &fds, NULL, NULL, NULL) < 0)
            std::cout<<"elo"<<std::endl;
        if (FD_ISSET(g_socket_fd, &fds))
            accept_new_user(&users, g_socket_fd);
        handle_clients(&fds, users);
    }

}

int main(){
    server s = server();
    s.begin_server(1234);

}