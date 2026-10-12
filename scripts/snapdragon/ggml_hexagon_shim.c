/*
 * Dynamic entrypoint shim for Qualcomm Hexagon NPU backend in llama.cpp.
 * Bridges Qualcomm's WHQL-signed core backend (GGML_BACKEND_API_VERSION 2)
 * to upstream llama.cpp's dynamic loader (GGML_BACKEND_API_VERSION 3).
 */

#include <windows.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "ggml-backend-impl.h"

// v2 buffer type interface as exported by Qualcomm WHQL core library
struct ggml_backend_buffer_type_i_v2 {
    const char *          (*get_name)      (ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*alloc_buffer)  (ggml_backend_buffer_type_t buft, size_t size);
    size_t                (*get_alignment) (ggml_backend_buffer_type_t buft);
    size_t                (*get_max_size)  (ggml_backend_buffer_type_t buft);
    size_t                (*get_alloc_size)(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
    bool                  (*is_host)       (ggml_backend_buffer_type_t buft);
};

struct ggml_backend_buffer_type_v2 {
    struct ggml_backend_buffer_type_i_v2 iface;
    ggml_backend_dev_t device;
    void * context;
};

typedef void* (*reg_fn_t)(void);

static ggml_backend_reg_t                  g_reg_v2       = NULL;
static ggml_backend_dev_t                  g_dev_v2       = NULL;
static struct ggml_backend_buffer_type_v2 * g_buft_v2     = NULL;
static struct ggml_backend_buffer_type_v2 * g_host_buft_v2 = NULL;

static struct ggml_backend_buffer_type     g_shim_buft;
static struct ggml_backend_buffer_type     g_shim_host_buft;
static struct ggml_backend_device          g_shim_dev;
static struct ggml_backend_reg             g_shim_reg;

// Buffer type callbacks (v3 layout)
static const char * shim_buft_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    if (g_buft_v2 && g_buft_v2->iface.get_name) {
        return g_buft_v2->iface.get_name((ggml_backend_buffer_type_t) g_buft_v2);
    }
    return "Hexagon";
}

static ggml_backend_buffer_t shim_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (!g_buft_v2 || !g_buft_v2->iface.alloc_buffer) {
        return NULL;
    }
    ggml_backend_buffer_t buf = g_buft_v2->iface.alloc_buffer((ggml_backend_buffer_type_t) g_buft_v2, size);
    if (buf) {
        buf->buft = buft;
    }
    return buf;
}

static size_t shim_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    if (g_buft_v2 && g_buft_v2->iface.get_alignment) {
        return g_buft_v2->iface.get_alignment((ggml_backend_buffer_type_t) g_buft_v2);
    }
    return 128;
}

static size_t shim_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    if (g_buft_v2 && g_buft_v2->iface.get_max_size) {
        return g_buft_v2->iface.get_max_size((ggml_backend_buffer_type_t) g_buft_v2);
    }
    return (size_t)-1;
}

static size_t shim_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    (void)buft;
    if (g_buft_v2 && g_buft_v2->iface.get_alloc_size) {
        return g_buft_v2->iface.get_alloc_size((ggml_backend_buffer_type_t) g_buft_v2, tensor);
    }
    return 0;
}

static bool shim_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    if (g_buft_v2 && g_buft_v2->iface.is_host) {
        return g_buft_v2->iface.is_host((ggml_backend_buffer_type_t) g_buft_v2);
    }
    return false;
}

// Host buffer callbacks
static const char * shim_host_buft_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    if (g_host_buft_v2 && g_host_buft_v2->iface.get_name) {
        return g_host_buft_v2->iface.get_name((ggml_backend_buffer_type_t) g_host_buft_v2);
    }
    return "Hexagon_Host";
}

static ggml_backend_buffer_t shim_host_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (!g_host_buft_v2 || !g_host_buft_v2->iface.alloc_buffer) {
        return NULL;
    }
    ggml_backend_buffer_t buf = g_host_buft_v2->iface.alloc_buffer((ggml_backend_buffer_type_t) g_host_buft_v2, size);
    if (buf) {
        buf->buft = buft;
    }
    return buf;
}

// Device callbacks
static const char * shim_dev_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    return g_dev_v2->iface.get_name(g_dev_v2);
}

static const char * shim_dev_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return g_dev_v2->iface.get_description(g_dev_v2);
}

static void shim_dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void)dev;
    g_dev_v2->iface.get_memory(g_dev_v2, free, total);
}

static enum ggml_backend_dev_type shim_dev_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return g_dev_v2->iface.get_type(g_dev_v2);
}

static void shim_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    (void)dev;
    g_dev_v2->iface.get_props(g_dev_v2, props);
}

static ggml_backend_t shim_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void)dev;
    ggml_backend_t backend = g_dev_v2->iface.init_backend(g_dev_v2, params);
    if (backend) {
        backend->device = &g_shim_dev;
    }
    return backend;
}

static ggml_backend_buffer_type_t shim_dev_get_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return &g_shim_buft;
}

static ggml_backend_buffer_type_t shim_dev_get_host_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    if (g_host_buft_v2) {
        return &g_shim_host_buft;
    }
    return NULL;
}

static ggml_backend_buffer_t shim_dev_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    (void)dev;
    if (g_dev_v2->iface.buffer_from_host_ptr) {
        ggml_backend_buffer_t buf = g_dev_v2->iface.buffer_from_host_ptr(g_dev_v2, ptr, size, max_tensor_size);
        if (buf) {
            buf->buft = &g_shim_buft;
        }
        return buf;
    }
    return NULL;
}

static bool shim_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;
    return g_dev_v2->iface.supports_op(g_dev_v2, op);
}

static bool shim_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    if (buft == &g_shim_buft) {
        buft = (ggml_backend_buffer_type_t) g_buft_v2;
    } else if (buft == &g_shim_host_buft) {
        buft = (ggml_backend_buffer_type_t) g_host_buft_v2;
    }
    return g_dev_v2->iface.supports_buft(g_dev_v2, buft);
}

static bool shim_dev_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;
    return g_dev_v2->iface.offload_op ? g_dev_v2->iface.offload_op(g_dev_v2, op) : false;
}

static ggml_backend_event_t shim_dev_event_new(ggml_backend_dev_t dev) {
    (void)dev;
    return g_dev_v2->iface.event_new ? g_dev_v2->iface.event_new(g_dev_v2) : NULL;
}

static void shim_dev_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    (void)dev;
    if (g_dev_v2->iface.event_free) {
        g_dev_v2->iface.event_free(g_dev_v2, event);
    }
}

static void shim_dev_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    (void)dev;
    if (g_dev_v2->iface.event_synchronize) {
        g_dev_v2->iface.event_synchronize(g_dev_v2, event);
    }
}

// Registry callbacks
static const char * shim_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return g_reg_v2->iface.get_name(g_reg_v2);
}

static size_t shim_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return g_reg_v2->iface.get_device_count(g_reg_v2);
}

static ggml_backend_dev_t shim_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg;
    (void)index;
    return &g_shim_dev;
}

static void * shim_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void)reg;
    return g_reg_v2->iface.get_proc_address ? g_reg_v2->iface.get_proc_address(g_reg_v2, name) : NULL;
}

static const struct ggml_backend_buffer_type_i g_buft_i_v3 = {
    /* .get_name         = */ shim_buft_get_name,
    /* .alloc_buffer     = */ shim_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ NULL,
    /* .get_alignment    = */ shim_buft_get_alignment,
    /* .get_max_size     = */ shim_buft_get_max_size,
    /* .get_alloc_size   = */ shim_buft_get_alloc_size,
    /* .get_alloc_size_n = */ NULL,
    /* .is_host          = */ shim_buft_is_host,
};

static const struct ggml_backend_buffer_type_i g_host_buft_i_v3 = {
    /* .get_name         = */ shim_host_buft_get_name,
    /* .alloc_buffer     = */ shim_host_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ NULL,
    /* .get_alignment    = */ shim_buft_get_alignment,
    /* .get_max_size     = */ shim_buft_get_max_size,
    /* .get_alloc_size   = */ shim_buft_get_alloc_size,
    /* .get_alloc_size_n = */ NULL,
    /* .is_host          = */ shim_buft_is_host,
};

static const struct ggml_backend_device_i g_dev_i_v3 = {
    /* .get_name             = */ shim_dev_get_name,
    /* .get_description      = */ shim_dev_get_description,
    /* .get_memory           = */ shim_dev_get_memory,
    /* .get_type             = */ shim_dev_get_type,
    /* .get_props            = */ shim_dev_get_props,
    /* .init_backend         = */ shim_dev_init_backend,
    /* .get_buffer_type      = */ shim_dev_get_buffer_type,
    /* .get_host_buffer_type = */ shim_dev_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ shim_dev_buffer_from_host_ptr,
    /* .supports_op          = */ shim_dev_supports_op,
    /* .supports_buft        = */ shim_dev_supports_buft,
    /* .offload_op           = */ shim_dev_offload_op,
    /* .event_new            = */ shim_dev_event_new,
    /* .event_free           = */ shim_dev_event_free,
    /* .event_synchronize    = */ shim_dev_event_synchronize,
};

static const struct ggml_backend_reg_i g_reg_i_v3 = {
    /* .get_name             = */ shim_reg_get_name,
    /* .get_device_count     = */ shim_reg_get_device_count,
    /* .get_device           = */ shim_reg_get_device,
    /* .get_proc_address     = */ shim_reg_get_proc_address,
};

__declspec(dllexport) void * ggml_backend_init(void);
__declspec(dllexport) int    ggml_backend_score(void);

__declspec(dllexport) int ggml_backend_score(void) {
    return 100;
}

__declspec(dllexport) void * ggml_backend_init(void) {
    if (g_reg_v2) {
        return &g_shim_reg;
    }

    HMODULE hSelf = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCSTR) ggml_backend_init, &hSelf);

    char path[MAX_PATH];
    DWORD n = GetModuleFileNameA(hSelf, path, MAX_PATH);
    if (n == 0) {
        return NULL;
    }

    // Strip binary filename to obtain directory
    char * lastSlash = NULL;
    for (char * p = path; *p; p++) {
        if (*p == '\\' || *p == '/') {
            lastSlash = p;
        }
    }
    if (!lastSlash) {
        return NULL;
    }
    *(lastSlash + 1) = '\0';

    // Target the core Qualcomm DLL in the same folder
    char coreDll[MAX_PATH];
    char * src = path;
    char * dst = coreDll;
    while (*src) {
        *dst++ = *src++;
    }
    const char * sub = "ggml-hexagon-core.dll";
    while (*sub) {
        *dst++ = *sub++;
    }
    *dst = '\0';

    // Load with LOAD_WITH_ALTERED_SEARCH_PATH so dependent DLLs are located
    HMODULE hCore = LoadLibraryExA(coreDll, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!hCore) {
        return NULL;
    }

    reg_fn_t fn = (reg_fn_t) GetProcAddress(hCore, "ggml_backend_hexagon_reg");
    if (!fn) {
        return NULL;
    }

    g_reg_v2 = (ggml_backend_reg_t) fn();
    if (!g_reg_v2) {
        return NULL;
    }

    g_dev_v2 = g_reg_v2->iface.get_device(g_reg_v2, 0);
    if (!g_dev_v2) {
        return NULL;
    }

    g_buft_v2 = (struct ggml_backend_buffer_type_v2 *) g_dev_v2->iface.get_buffer_type(g_dev_v2);
    if (g_dev_v2->iface.get_host_buffer_type) {
        g_host_buft_v2 = (struct ggml_backend_buffer_type_v2 *) g_dev_v2->iface.get_host_buffer_type(g_dev_v2);
    }

    // Initialize v3 structures
    g_shim_buft.iface   = g_buft_i_v3;
    g_shim_buft.device  = &g_shim_dev;
    g_shim_buft.context = g_buft_v2;

    if (g_host_buft_v2) {
        g_shim_host_buft.iface   = g_host_buft_i_v3;
        g_shim_host_buft.device  = &g_shim_dev;
        g_shim_host_buft.context = g_host_buft_v2;
    }

    g_shim_dev.iface   = g_dev_i_v3;
    g_shim_dev.reg     = &g_shim_reg;
    g_shim_dev.context = g_dev_v2->context;

    g_shim_reg.api_version = GGML_BACKEND_API_VERSION; // 3
    g_shim_reg.iface       = g_reg_i_v3;
    g_shim_reg.context     = g_reg_v2->context;

    return &g_shim_reg;
}
