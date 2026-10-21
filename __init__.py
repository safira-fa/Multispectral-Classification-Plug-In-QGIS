# -*- coding: utf-8 -*-
def classFactory(iface):
    from .plugin import MultispectralClassifierPlugin
    return MultispectralClassifierPlugin(iface)