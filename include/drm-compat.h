/* Android compatibility header for DRM definitions */
#ifndef DRM_COMPAT_H
#define DRM_COMPAT_H

#include <linux/ioctl.h>
#include <linux/types.h>

/* DRM command base */
#ifndef DRM_COMMAND_BASE
#define DRM_COMMAND_BASE                0x40
#endif

/* DRM ioctl macros */
#ifndef DRM_IOC_READ
#define DRM_IOC_READ                    _IOC(_IOC_READ, 'd', 0x00, 0x00)
#endif

#ifndef DRM_IOC_WRITE
#define DRM_IOC_WRITE                   _IOC(_IOC_WRITE, 'd', 0x00, 0x00)
#endif

#ifndef DRM_IOC_READWRITE
#define DRM_IOC_READWRITE               _IOC(_IOC_READ|_IOC_WRITE, 'd', 0x00, 0x00)
#endif

#define DRM_IOC(nr)                     _IOC(DRM_IOC_READWRITE, 'd', nr, 0)
#define DRM_IOW(nr, type)               _IOW('d', nr, type)
#define DRM_IOR(nr, type)               _IOR('d', nr, type)
#define DRM_IOWR(nr, type)              _IOWR('d', nr, type)

/* DRM version ioctl */
#define DRM_IOCTL_VERSION               DRM_IOR(0x00, struct drm_version)

/* DRM version structure */
struct drm_version {
    int version_major;
    int version_minor;
    int version_patchlevel;
    __u32 name_len;
    char *name;
    __u32 date_len;
    char *date;
    __u32 desc_len;
    char *desc;
};

#endif /* DRM_COMPAT_H */

