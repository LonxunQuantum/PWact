
from active_learning.init_bulk.relax import Relax
from active_learning.init_bulk.duplicate_scale import duplicate_scale, do_pertub
from active_learning.init_bulk.aimd import AIMD

from active_learning.user_input.init_bulk_input import InitBulkParam
from active_learning.user_input.resource import Resource

def init_bulk_run(resource: Resource, input_param:InitBulkParam):
    #1. do relax
    relax = Relax(resource, input_param)
    if not relax.check_work_done():
        # make relax work dir
        relax.make_relax_work()
        # do relax jobs
        relax.do_relax_jobs()
        # do post process
        relax.do_post_process()
        
    # do super cell and scale
    duplicate_scale(resource, input_param)
    # do pertub
    do_pertub(resource, input_param)
    # do scf
    aimd = AIMD(resource, input_param)
    aimd.make_scf_work()
    aimd.do_scf_jobs()
    
    # do collection
    # delete no use files
    
# def check_work_state():
    