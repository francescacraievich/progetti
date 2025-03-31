#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include<dirent.h>
#include<stdint.h>
#include<sys/stat.h>
#include <pwd.h>
#include <grp.h>

void cerca (char *path);
void print (struct stat file, char *path);

int main(int argc, char *argv[]){
	
	struct stat info_file;
	if (argc!=2){
	printf("specifica un path\n");
	return 0;
	if (stat(argv[1], &info_file)<0){
        printf("Impossibile leggere le informazioni del file\n");
        exit(1);
        }
        return 0;
	}
	stat(argv[1], &info_file);
	cerca(argv[1]);
	print(info_file,argv[1]);
	
}

void print(struct stat file, char *path){
        struct passwd *pwd;
        struct group *grp;
        pwd = getpwuid(file.st_uid);
	grp = getgrgid(file.st_gid);
	
	printf("\nNode: %s \n", path);
	printf("\tInode: %d \n", file.st_ino);
	printf("\tType: ");
	switch(file.st_mode & S_IFMT){
        case   S_IFLNK:     printf("symbolic link\n"); break;
        case   S_IFREG:   printf("file\n"); break;
        case   S_IFDIR:   printf("directory\n"); break;
        case   S_IFIFO:   printf("FIFO\n"); break;
        default: 	printf("other\n");
	};
	
	printf("\tSize: %d\n", file.st_size);
	printf("\tOwner: %d %s \n", file.st_uid, pwd->pw_name);
	printf("\tGroup: %d %s \n", file.st_gid, grp->gr_name);
	}
	
void cerca(char *path){
	DIR *directory;
	struct dirent *file;
	struct stat info_file;
 	directory = opendir(path);
 	while (((file=readdir(directory)))!= NULL){
 	char newpath[100]="";
 	if ((strncmp(file -> d_name, ".", 1))==0 || strncmp(file->d_name, "..", 2)==0){
 	continue;
 	}
 	
 	strcat(newpath, path);
 	strcat (newpath, "/");
	strcat (newpath, file -> d_name);
	stat(newpath, &info_file);
	print(info_file, newpath);
	
 	if (S_ISDIR(info_file.st_mode)) {
	if ((strncmp(file -> d_name, ".", 1))!=0 && strncmp(file->d_name, "..", 2)!=0){
	cerca(newpath);
 	}
 	}
 	}
closedir(directory);
 }

