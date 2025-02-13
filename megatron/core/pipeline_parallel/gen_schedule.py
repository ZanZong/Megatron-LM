from enum import Enum, auto
from typing import List

class Ops(Enum):
    forward = auto()
    backward = auto()
    send_forward = auto()
    recv_forward = auto()
    send_backward = auto()
    recv_backward = auto()
    send_forward_recv_backward = auto()
    send_backward_recv_forward = auto()
    synchronize = auto()
    fw_end = auto()
    bw_end = auto()
    
class PipelineInstruction():
    def __init__(self, op_type, model_chunk_id=None, micro_batch_id=None, up_or_down=None) -> None:
        self.op_type = op_type
        self.model_chunk_id = model_chunk_id
        if isinstance(micro_batch_id, tuple):
            self.micro_batch_id, self.recv_micro_batch_id = micro_batch_id
        else:
            self.micro_batch_id = micro_batch_id
        self.up_or_down = up_or_down
    
    def __str__(self) -> str:
        modal = 'Text' if self.model_chunk_id == 1 else 'Vision'
        return "{}-{}: {}" \
                .format(modal, self.micro_batch_id, self.op_type)

class Schedules3:
    def __init__(self) -> None:
        # Save ordered instructions
        self._inst = []
    
    def generate(self, rank, pipeline_parallel_world_size, num_microbatches, modal_ratio=2, batched_backward_k=None) -> List[Ops]:
        """Computation schedules of bidirectional pipeline cannot be launched in order,
        because the opposite send ops will block the schedule. We rearrange the order
        to make the op list can execute one-by-one.
        
        Bidirectional pipeline impl. The model chuck order is [image, text].
        """
        self.free_schedule()
        num_warmup_microbatches = min(pipeline_parallel_world_size - rank - 1, num_microbatches)
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches
        next_rank_num_microbatches_remaining = num_microbatches - min(pipeline_parallel_world_size - rank - 2, num_microbatches)
        # print(f"num_warmup_microbatches={num_warmup_microbatches}")
        # print(f"next_rank_num_microbatches_remaining={next_rank_num_microbatches_remaining}")
        get_up_pipeline_warmup_batches = lambda rank : max(0, min(int(rank * modal_ratio) - \
                                    (pipeline_parallel_world_size - 1 - rank), num_microbatches))
        up_pipeline_warmup_batches = get_up_pipeline_warmup_batches(rank)
        up_pipeline_next_stage_warmup_batches = get_up_pipeline_warmup_batches(rank - 1)
        up_pipeline_remaining_batches = num_microbatches - up_pipeline_warmup_batches


        def get_hang_before_steady(local_rank):
            for rank_i in list(range(pipeline_parallel_world_size))[::-1]:
                if get_up_pipeline_warmup_batches(rank_i) == 0:
                    hang_before_steady_first_full_rank = rank
                    break
            if local_rank > hang_before_steady_first_full_rank:
                hang_before_steady = 0
            else:
                hang_before_steady = hang_before_steady_first_full_rank - local_rank
            return hang_before_steady
        
        def get_early_backward_num(local_rank):
            # Calculate how many early backward can be scheduled for up pipeline
            V_space_len = (pipeline_parallel_world_size - local_rank - 1) * 2 * modal_ratio
            A_space_len = 3 * local_rank # light-weight forward and backward
            max_backward_num_wrt_constrastive_loss = max(0, int((pipeline_parallel_world_size - 2 - local_rank) * 2 / 3 + 1))
            max_backward_num_wrt_space = max(0, (V_space_len - A_space_len - \
                        (num_microbatches - get_up_pipeline_warmup_batches(local_rank)) - get_hang_before_steady(local_rank)) // 2)
            # Three factors determine valid early backward num
            #   1. contrastive loss output of another side pipeline.
            #   2. max backward number to inserted.
            #   3. how many microbatches needed.
            return min(max_backward_num_wrt_constrastive_loss, max_backward_num_wrt_space, num_microbatches)
        early_backward_num_all = [get_early_backward_num(i) for i in range(pipeline_parallel_world_size)]
        early_backward_num_all.append(0) # ensure the last rank can access rank+1
        early_backward_num = early_backward_num_all[rank]
        print(f"Rank{rank} -- num_micro_batches={num_microbatches}, \
              text_warmup_batches={up_pipeline_warmup_batches}, \
                text_remaining_batches={up_pipeline_remaining_batches} \
                    vision_warmup_batches={num_warmup_microbatches}, \
                        early_backward={early_backward_num}, \
                          remaining_batch={num_microbatches_remaining}"  , flush=True)
        # exit()
        # print(f"early_backward_num_all={early_backward_num_all}")
        for rank_i, backward_num in enumerate(early_backward_num_all):
            if backward_num == 0:
                early_backward_last_rank = max(0, rank_i - 1)
                break
        send_up_pended = []
        send_down_pended = []
        # for text warmup
        for i in range(up_pipeline_warmup_batches):
            self._inst.append(PipelineInstruction(Ops.recv_forward, 1, i, "up"))
            self._inst.append(PipelineInstruction(Ops.forward, 1, i, "up"))
            pp_inst = PipelineInstruction(Ops.send_forward, 1, i, "up")
            if i < up_pipeline_next_stage_warmup_batches:
                self._inst.append(pp_inst)
                self._inst.append(PipelineInstruction(Ops.fw_end, 1, i, "up"))
            else:
                send_up_pended.append(pp_inst)
        # print(f"send_up_pended after text warmup={send_up_pended}")
        # Run warmup forward passes. Vision forward
        for i in range(num_warmup_microbatches):
            self._inst.append(PipelineInstruction(Ops.recv_forward, 0, i, "down"))
            
            # Run vision warmup forward    
            self._inst.append(PipelineInstruction(Ops.forward, 0, i, "down"))
            # last warmup batch in second last stage directly send, else pending
            pp_inst = PipelineInstruction(Ops.send_forward, 0, i, "down")
            if i == num_warmup_microbatches - 1 and next_rank_num_microbatches_remaining > 0:
                send_down_pended.append(pp_inst)
                # If is the last warmup forward, send up pipeline's out tensors.
                if len(send_up_pended) > 0:
                    send_up_inst = send_up_pended.pop(0)
                    self._inst.append(send_up_inst)
                    self._inst.append(PipelineInstruction(Ops.fw_end, 1, send_up_inst.micro_batch_id, "up"))
            else:
                self._inst.append(pp_inst)
                self._inst.append(PipelineInstruction(Ops.fw_end, 0, i, "down"))

        early_backward_batches = list(range(int(early_backward_num)))
        send_up_backward_pend = []
        # Run up pipeline remaining forward. Text Remaining
        for i in range(up_pipeline_remaining_batches):
            micro_batch_id = i + up_pipeline_warmup_batches
            self._inst.append(PipelineInstruction(Ops.recv_forward, 1, micro_batch_id, "up"))
            # if len(send_up_backward_pend) > 0:
            #     self._inst.append(send_up_backward_pend.pop(0))
            self._inst.append(PipelineInstruction(Ops.forward, 1, micro_batch_id, "up"))
            pp_inst = PipelineInstruction(Ops.send_forward, 1, micro_batch_id, "up")
            # Send pended up forward first
            if len(send_up_pended) > 0:
                send_up_inst = send_up_pended.pop(0)
                send_up_id = send_up_inst.micro_batch_id
                self._inst.append(send_up_inst)
                self._inst.append(PipelineInstruction(Ops.fw_end, 1, send_up_id, "up"))
                send_up_pended.append(pp_inst)
            else:
                self._inst.append(pp_inst)
                self._inst.append(PipelineInstruction(Ops.fw_end, 1, micro_batch_id, "up"))
        
        while len(send_up_pended) > 0:
            send_up_inst = send_up_pended.pop(0)
            send_up_id = send_up_inst.micro_batch_id
            self._inst.append(send_up_inst)
            self._inst.append(PipelineInstruction(Ops.fw_end, 1, send_up_id, "up"))
            
        # Insert early backward
        while len(early_backward_batches) != 0:
            backward_batch_id = early_backward_batches.pop(0)
            # print(f"pop backward batch {backward_batch_id}")
            self._inst.append(PipelineInstruction(Ops.recv_backward, 1, backward_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.backward, 1, backward_batch_id, "up"))
            if rank < early_backward_last_rank and backward_batch_id < early_backward_num_all[rank + 1]:
                self._inst.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.bw_end, 1, backward_batch_id, "up"))
            else:
                send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, backward_batch_id, "up"))
                
        # while len(send_up_backward_pend) > 0:
        #     self._inst.append(send_up_backward_pend.pop(0))
        
        # Insert send forward down for steady phase
        if len(send_down_pended) > 0:
            send_down_inst = send_down_pended.pop(0)
            send_down_id = send_down_inst.micro_batch_id
            self._inst.append(send_down_inst)
            self._inst.append(PipelineInstruction(Ops.fw_end, 0, send_down_id, "down"))


        if num_microbatches_remaining > 0:
            # recv the first steady phase vision forward
            self._inst.append(PipelineInstruction(Ops.recv_forward, 0, num_warmup_microbatches, "down"))
        # Run 1F1B in steady state.
        vision_backward_micro_batch_id = 0
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)
            micro_batch_id = num_warmup_microbatches + i
            next_micro_batch_id = micro_batch_id + 1
            self._inst.append(PipelineInstruction(Ops.forward, 0, micro_batch_id, "down"))
            self._inst.append(PipelineInstruction(Ops.send_forward_recv_backward, 0, (micro_batch_id, vision_backward_micro_batch_id), "down"))
            self._inst.append(PipelineInstruction(Ops.fw_end, 0, micro_batch_id, "down"))
            
            self._inst.append(PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down"))
            if last_iteration:
                self._inst.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
            else:
                self._inst.append(PipelineInstruction(Ops.send_backward_recv_forward, 0, (vision_backward_micro_batch_id, micro_batch_id + 1), "down"))
            
            self._inst.append(PipelineInstruction(Ops.bw_end, 0, vision_backward_micro_batch_id, "down"))
            vision_backward_micro_batch_id += 1
        
        # text cool down
        text_backward_micro_batch_id = 0 + early_backward_num
        # Cool down
        for i in range(num_warmup_microbatches):
            # Backward text cool down
            # num_microbatches_remaining > 0 considers case pp=8, mb_num=4
            if num_microbatches_remaining > 0:
                if text_backward_micro_batch_id < num_microbatches:
                    self._inst.append(PipelineInstruction(Ops.recv_backward, 1, text_backward_micro_batch_id, "up"))
                    self._inst.append(PipelineInstruction(Ops.backward, 1, text_backward_micro_batch_id, "up"))
                    send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, text_backward_micro_batch_id, "up"))
                    text_backward_micro_batch_id += 1
            # Backward vision cool down
            self._inst.append(PipelineInstruction(Ops.recv_backward, 0, vision_backward_micro_batch_id, "down"))
            if num_microbatches_remaining > 0:
                if len(send_up_backward_pend) > 0:
                    send_up_inst = send_up_backward_pend.pop(0)
                    send_up_id = send_up_inst.micro_batch_id
                    self._inst.append(send_up_inst)
                    self._inst.append(PipelineInstruction(Ops.bw_end, 1, send_up_id, "up"))

            self._inst.append(PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down"))
            self._inst.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
            self._inst.append(PipelineInstruction(Ops.bw_end, 0, vision_backward_micro_batch_id, "down"))
            if num_microbatches_remaining == 0: # 8stage 4micro batches no steady 1F1B
                if text_backward_micro_batch_id < num_microbatches:
                    self._inst.append(PipelineInstruction(Ops.recv_backward, 1, text_backward_micro_batch_id, "up"))
                    self._inst.append(PipelineInstruction(Ops.backward, 1, text_backward_micro_batch_id, "up"))
                    send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, text_backward_micro_batch_id, "up"))
                    text_backward_micro_batch_id += 1
                if len(send_up_backward_pend) > 0:
                    send_up_inst = send_up_backward_pend.pop(0)
                    send_up_id = send_up_inst.micro_batch_id
                    self._inst.append(send_up_inst)
                    self._inst.append(PipelineInstruction(Ops.bw_end, 1, send_up_id, "up"))
            vision_backward_micro_batch_id += 1
        
        # remaining text cool down
        while len(send_up_backward_pend) > 0:
            send_up_inst = send_up_backward_pend.pop(0)
            send_up_id = send_up_inst.micro_batch_id
            self._inst.append(send_up_inst)
            self._inst.append(PipelineInstruction(Ops.bw_end, 1, send_up_id, "up"))
        while text_backward_micro_batch_id < num_microbatches:
            self._inst.append(PipelineInstruction(Ops.recv_backward, 1, text_backward_micro_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.backward, 1, text_backward_micro_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.send_backward, 1, text_backward_micro_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.bw_end, 1, text_backward_micro_batch_id, "up"))
            text_backward_micro_batch_id += 1
            
        return self._inst
            
    def format_schedule(self):
        for sche in self._inst:
            vision_or_text = "vision" if sche.model_chunk_id == 0 else "text"
            if hasattr(sche, "recv_micro_batch_id"):
                recv_micro_batch_id = sche.recv_micro_batch_id
                print(f"{vision_or_text}, {sche.op_type}, send micro_batch_id={sche.micro_batch_id}, recv micro_batch_id={recv_micro_batch_id}")
            else:    
                print(f"{vision_or_text}, {sche.op_type}, micro_batch_id={sche.micro_batch_id}")

    def free_schedule(self):
        self._inst = []

class Schedules4():
    def __init__(self) -> None:
        # Save ordered instructions
        self._inst = []
        self.prev_fw_bw_inst = None

    def format_schedule(self):
        for sche in self._inst:
            vision_or_text = "vision" if sche.model_chunk_id == 0 else "text"
            if hasattr(sche, "recv_micro_batch_id"):
                recv_micro_batch_id = sche.recv_micro_batch_id
                print(f"{vision_or_text}, {sche.op_type}, send micro_batch_id={sche.micro_batch_id}, recv micro_batch_id={recv_micro_batch_id}")
            else:    
                print(f"{vision_or_text}, {sche.op_type}, micro_batch_id={sche.micro_batch_id}")

    def free_schedule(self):
        self._inst = []

    def generate(self, rank, pipeline_parallel_world_size, num_microbatches, real_modal_ratio, intensive_backward_k=None, fuse_light_factor=1, duplicate_factor=1) -> List[Ops]:
        """Split vision backward and insert text backward
        
        Bidirectional pipeline impl. The model chuck order is [image, text].
        """    
        self.free_schedule() 
        up_ranks = []
        up_pipeline_size = pipeline_parallel_world_size//duplicate_factor
        for _ in range(duplicate_factor):
            up_ranks.extend([j for j in range(up_pipeline_size)][::-1])
        num_warmup_microbatches = min(pipeline_parallel_world_size - rank - 1, num_microbatches)
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches
        modal_ratio = real_modal_ratio # Assume fusing light modal will reduce the modality ratio accordingly. TODO: should be unlinear.
        # print(f"REAL MODAL RATIO={modal_ratio}", flush=True)
        next_rank_num_microbatches_remaining = num_microbatches - min(pipeline_parallel_world_size - rank - 2, num_microbatches)
        light_num_microbatches = num_microbatches // fuse_light_factor // duplicate_factor
        light_pipeline_id = rank // (pipeline_parallel_world_size // duplicate_factor)
        # up_pipeline_parallel_world_size = pipeline_parallel_world_size // duplicate_factor
        get_up_pipeline_warmup_batches = lambda rank : max(0, min(int(rank * modal_ratio) - \
                    up_ranks[rank], light_num_microbatches))
        up_pipeline_warmup_batches = get_up_pipeline_warmup_batches(rank)
        up_pipeline_next_stage_warmup_batches = get_up_pipeline_warmup_batches(rank - 1)
        up_pipeline_remaining_batches = light_num_microbatches - up_pipeline_warmup_batches
        self.prev_fw_bw_inst = None

        def get_hang_before_steady(local_rank):
            for rank_i in list(range(pipeline_parallel_world_size))[::-1]:
                if get_up_pipeline_warmup_batches(rank_i) == 0:
                    hang_before_steady_first_full_rank = rank_i
                    break
            return max(0, hang_before_steady_first_full_rank - local_rank)
        
        def get_early_backward_num(local_rank):
            # Calculate how many early backward can be scheduled for up pipeline
            # formed a 'V' space
            V_space_len = (pipeline_parallel_world_size - local_rank - 1) * 2 * modal_ratio
            # formed a 'A' space between light-weight forward and backward
            A_space_len = 3 * local_rank
            max_backward_num_wrt_constrastive_loss = max(0, int((pipeline_parallel_world_size // duplicate_factor - 2 - local_rank) * 2 / 3 + 1))
            max_backward_num_wrt_space = max(0, (V_space_len - A_space_len - \
                        (light_num_microbatches - get_up_pipeline_warmup_batches(local_rank)) - get_hang_before_steady(local_rank)) // 2)
            # Three factors determine valid early backward num
            #   1. contrastive loss output of another side pipeline.
            #   2. max backward number to inserted.
            #   3. how many microbatches needed.
            return min(round(max_backward_num_wrt_constrastive_loss), round(max_backward_num_wrt_space), light_num_microbatches)
        early_backward_num_all = [0 for i in range(pipeline_parallel_world_size)]
        # early_backward_num_all = [get_early_backward_num(i) for i in range(pipeline_parallel_world_size)]
        early_backward_num_all.append(0) # ensure the last rank can access rank+1
        early_backward_num = early_backward_num_all[rank]
        # print(f"Rank{rank} up_rank{up_ranks[rank]} -- num_micro_batches={num_microbatches}, \
        #       text_warmup_batches={up_pipeline_warmup_batches}, \
        #         text_remaining_batches={up_pipeline_remaining_batches} \
        #             vision_warmup_batches={num_warmup_microbatches}, \
        #                 early_backward={early_backward_num}, \
        #                   remaining_batch={num_microbatches_remaining}", flush=True)
        # print(f"early_backward_num_all={early_backward_num_all}")
        for rank_i, backward_num in enumerate(early_backward_num_all):
            if backward_num == 0:
                early_backward_last_rank = max(0, rank_i - 1)
                break
        send_up_pended = []
        send_down_pended = []
        # for text warmup
        for i in range(up_pipeline_warmup_batches):
            light_mb_id = light_pipeline_id * light_num_microbatches + i
            self._inst.append(PipelineInstruction(Ops.recv_forward, 1, light_mb_id, "up"))
            self._inst.append(PipelineInstruction(Ops.forward, 1, light_mb_id, "up"))
            self.prev_fw_bw_inst = PipelineInstruction(Ops.forward, 1, light_mb_id, "up")
            pp_inst = PipelineInstruction(Ops.send_forward, 1, light_mb_id, "up")
            if i < up_pipeline_next_stage_warmup_batches:
                self._inst.append(pp_inst)
                self._inst.append(PipelineInstruction(Ops.fw_end, 1, light_mb_id, "up"))
            else:
                send_up_pended.append(pp_inst)
        # print(f"send_up_pended after text warmup={send_up_pended}")
        # Run warmup forward passes. Vision forward
        for i in range(num_warmup_microbatches):

            self._inst.append(PipelineInstruction(Ops.recv_forward, 0, i, "down")) 
            # Run vision warmup forward    
            self._inst.append(PipelineInstruction(Ops.forward, 0, i, "down"))
            self.prev_fw_bw_inst = PipelineInstruction(Ops.forward, 0, i, "down")

            # last warmup batch in second last stage directly send, else pending
            pp_inst = PipelineInstruction(Ops.send_forward, 0, i, "down")
            if i == num_warmup_microbatches - 1 and next_rank_num_microbatches_remaining > 0:
                send_down_pended.append(pp_inst)
                # If is the last warmup forward, send up pipeline's out tensors.
                if len(send_up_pended) > 0:
                    send_up_inst = send_up_pended.pop(0)
                    self._inst.append(send_up_inst)
                    self._inst.append(PipelineInstruction(Ops.fw_end, 1, send_up_inst.micro_batch_id, "up"))
            else:
                self._inst.append(pp_inst)
                self._inst.append(PipelineInstruction(Ops.fw_end, 0, i, "down"))

        early_backward_batches = list(range(int(early_backward_num)))
        # print(f"early_backward_bacth={early_backward_batches}")
        send_up_backward_pend = []
        # Run up pipeline remaining forward. Text Remaining
        for i in range(up_pipeline_remaining_batches):
            micro_batch_id = i + up_pipeline_warmup_batches
            light_mb_id = light_pipeline_id * light_num_microbatches + micro_batch_id
            
            self._inst.append(PipelineInstruction(Ops.recv_forward, 1, light_mb_id, "up"))
            # if len(send_up_backward_pend) > 0:
            #     self._inst.append(send_up_backward_pend.pop(0))
            self._inst.append(PipelineInstruction(Ops.forward, 1, light_mb_id, "up"))
            self.prev_fw_bw_inst = PipelineInstruction(Ops.forward, 1, light_mb_id, "up")

            pp_inst = PipelineInstruction(Ops.send_forward, 1, light_mb_id, "up")
            # Send pended up forward first
            if len(send_up_pended) > 0:
                send_up_inst = send_up_pended.pop(0)
                send_up_id = send_up_inst.micro_batch_id
                self._inst.append(send_up_inst)
                self._inst.append(PipelineInstruction(Ops.fw_end, 1, send_up_id, "up"))
                send_up_pended.append(pp_inst)
            else:
                self._inst.append(pp_inst)
                self._inst.append(PipelineInstruction(Ops.fw_end, 1, pp_inst.micro_batch_id, "up"))
        
        while len(send_up_pended) > 0:
            send_up_inst = send_up_pended.pop(0)
            send_up_id = send_up_inst.micro_batch_id
            self._inst.append(send_up_inst)
            self._inst.append(PipelineInstruction(Ops.fw_end, 1, send_up_id, "up"))
            
        # Insert early backward
        while len(early_backward_batches) != 0:
            backward_batch_id = early_backward_batches.pop(0)
            real_bw_batch_id = light_pipeline_id * light_num_microbatches + backward_batch_id
            # print(f"pop backward batch {backward_batch_id}")

            self._inst.append(PipelineInstruction(Ops.recv_backward, 1, real_bw_batch_id, "up"))
            self._inst.append(PipelineInstruction(Ops.backward, 1, real_bw_batch_id, "up"))
            self.prev_fw_bw_inst = PipelineInstruction(Ops.backward, 1, real_bw_batch_id, "up")

            if up_ranks[rank] == 0 or (rank < early_backward_last_rank and backward_batch_id < early_backward_num_all[rank + 1]):
                self._inst.append(PipelineInstruction(Ops.send_backward, 1, real_bw_batch_id, "up"))
                self._inst.append(PipelineInstruction(Ops.bw_end, 1, real_bw_batch_id, "up"))
            elif up_ranks[rank] != 0: # 当前rank是text的第一个rank
                send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, real_bw_batch_id, "up"))

        # Insert send forward down for steady phase
        if len(send_down_pended) > 0:
            send_down_inst = send_down_pended.pop(0)
            send_down_id = send_down_inst.micro_batch_id
            self._inst.append(send_down_inst)
            self._inst.append(PipelineInstruction(Ops.fw_end, 0, send_down_id, "down"))

        if num_microbatches_remaining > 0:
            # recv the first steady phase vision forward
            self._inst.append(PipelineInstruction(Ops.recv_forward, 0, num_warmup_microbatches, "down"))
        # Save the last 1f1b output, which will be sent in split
        last_1f1b_backward_output = []
        # Run 1F1B in steady state.
        vision_backward_micro_batch_id = 0
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)
            micro_batch_id = num_warmup_microbatches + i
            next_micro_batch_id = micro_batch_id + 1
            self._inst.append(PipelineInstruction(Ops.forward, 0, micro_batch_id, "down"))
            self.prev_fw_bw_inst = PipelineInstruction(Ops.forward, 0, micro_batch_id, "down")
              
            self._inst.append(PipelineInstruction(Ops.send_forward_recv_backward, 0, (micro_batch_id, vision_backward_micro_batch_id), "down"))
            self._inst.append(PipelineInstruction(Ops.fw_end, 0, micro_batch_id, "down"))
            self._inst.append(PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down"))
            self.prev_fw_bw_inst = PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down")

            if last_iteration:
                # self._inst.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
                last_1f1b_backward_output.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
            else:
                self._inst.append(PipelineInstruction(Ops.send_backward_recv_forward, 0, (vision_backward_micro_batch_id, micro_batch_id + 1), "down"))
                self._inst.append(PipelineInstruction(Ops.bw_end, 0, vision_backward_micro_batch_id, "down"))
            

            vision_backward_micro_batch_id += 1
        
        # Cool down
        text_backward_micro_batch_id = 0 + early_backward_num
        # TODO auto-tune this number
        intensive_backward_k = light_num_microbatches // 2 if intensive_backward_k is None else intensive_backward_k
        # init one more for rank+1
        intensive_text_backward_num = [intensive_backward_k for _ in range(pipeline_parallel_world_size + 1)]
        for rank_i in range(pipeline_parallel_world_size):
            if early_backward_num_all[rank_i] == light_num_microbatches:
                intensive_text_backward_num[rank_i] = 0
            else:
                intensive_text_backward_num[rank_i] = min(light_num_microbatches - early_backward_num_all[rank_i], intensive_backward_k)
        # print(f"intensive_text_backward_num={intensive_text_backward_num}")
        done_intensive_text_backward = False
        # Backward text cool down
        for i in range(num_warmup_microbatches):
            # num_microbatches_remaining > 0 considers case pp=8, mb_num=4
            if num_microbatches_remaining > 0:
                # Even all text backward have been finished, still need to send backward for prev. rank
                if text_backward_micro_batch_id <= light_num_microbatches:
                    if not done_intensive_text_backward:
                        done_intensive_text_backward = True
                        # If next stage compute more during intensive backward, the current stage should have more early backwards.
                        # Send pended backward first.
                        if up_ranks[rank] != 0 and len(send_up_backward_pend) > 0:
                            for _ in range(intensive_text_backward_num[rank + 1] - intensive_text_backward_num[rank]):
                                send_up_inst = send_up_backward_pend.pop(0)
                                self._inst.append(send_up_inst)
                                self._inst.append(PipelineInstruction(Ops.bw_end, 1, send_up_inst.micro_batch_id, "up"))
                        # Batch some text backward after 1F1B
                        num_early_bw_pended = len(send_up_backward_pend)
                        # print(F"rank={rank}, intensive_text_backward_num[rank]={intensive_text_backward_num[rank]}")
                        if intensive_text_backward_num[rank] == 0:
                            # Send the last 1f1b vision backward
                            last_1f1b_bw_inst = last_1f1b_backward_output.pop()
                            self._inst.append(last_1f1b_bw_inst)
                            self._inst.append(PipelineInstruction(Ops.bw_end, 0, last_1f1b_bw_inst.micro_batch_id, "down"))
                        for j in range(intensive_text_backward_num[rank]):
                            light_mb_id = light_pipeline_id * light_num_microbatches + text_backward_micro_batch_id
                            self._inst.append(PipelineInstruction(Ops.recv_backward, 1, light_mb_id, "up"))
                            if j == intensive_text_backward_num[rank] - 1:
                                # Send the last 1f1b vision backward after the last intensive backward
                                last_1f1b_bw_inst = last_1f1b_backward_output.pop()
                                self._inst.append(last_1f1b_bw_inst)
                                self._inst.append(PipelineInstruction(Ops.bw_end, 0, last_1f1b_bw_inst.micro_batch_id, "down"))
                            self._inst.append(PipelineInstruction(Ops.backward, 1, light_mb_id, "up"))
                            self.prev_fw_bw_inst = PipelineInstruction(Ops.backward, 1, light_mb_id, "up")

                            inst_tobe_sent = PipelineInstruction(Ops.send_backward, 1, light_mb_id, "up")
                            # if j == intensive_text_backward_num[rank] - 1:
                            send_up_backward_pend.append(inst_tobe_sent)
                            send_up_inst = send_up_backward_pend.pop(0)
                            self._inst.append(send_up_inst)
                            self._inst.append(PipelineInstruction(Ops.bw_end, 1, send_up_inst.micro_batch_id, "up"))
                            text_backward_micro_batch_id += 1
                    elif text_backward_micro_batch_id < light_num_microbatches:
                        light_mb_id = light_pipeline_id * light_num_microbatches + text_backward_micro_batch_id
                        self._inst.append(PipelineInstruction(Ops.recv_backward, 1, light_mb_id, "up"))
                        self._inst.append(PipelineInstruction(Ops.backward, 1, light_mb_id, "up"))
                        self.prev_fw_bw_inst = PipelineInstruction(Ops.backward, 1, light_mb_id, "up")
                        send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, light_mb_id, "up"))
                        text_backward_micro_batch_id += 1
            # Backward vision cool down
            self._inst.append(PipelineInstruction(Ops.recv_backward, 0, vision_backward_micro_batch_id, "down"))
            if len(send_up_backward_pend) > 0:
                send_up_inst = send_up_backward_pend.pop(0)
                send_up_id = send_up_inst.micro_batch_id
                self._inst.append(send_up_inst)
                self._inst.append(PipelineInstruction(Ops.bw_end, 1, send_up_id, "up"))
            self._inst.append(PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down"))
            self.prev_fw_bw_inst = PipelineInstruction(Ops.backward, 0, vision_backward_micro_batch_id, "down")
            self._inst.append(PipelineInstruction(Ops.send_backward, 0, vision_backward_micro_batch_id, "down"))
            self._inst.append(PipelineInstruction(Ops.bw_end, 0, vision_backward_micro_batch_id, "down"))
            if num_microbatches_remaining == 0: # 8stage 4micro batches no steady 1F1B
                if text_backward_micro_batch_id < light_num_microbatches:
                    light_mb_id = light_pipeline_id * light_num_microbatches + text_backward_micro_batch_id
                    self._inst.append(PipelineInstruction(Ops.recv_backward, 1, light_mb_id, "up"))
                    self._inst.append(PipelineInstruction(Ops.backward, 1, light_mb_id, "up"))
                    self.prev_fw_bw_inst = PipelineInstruction(Ops.backward, 1, light_mb_id, "up")

                    send_up_backward_pend.append(PipelineInstruction(Ops.send_backward, 1, light_mb_id, "up"))
                    text_backward_micro_batch_id += 1
                if len(send_up_backward_pend) > 0:
                    send_up_inst = send_up_backward_pend.pop(0)
                    send_up_id = send_up_inst.micro_batch_id
                    self._inst.append(send_up_inst)
                    self._inst.append(PipelineInstruction(Ops.bw_end, 1, send_up_id, "up"))
            vision_backward_micro_batch_id += 1
        
        # remaining text cool down, last rank of pipeline doesn't have warmup    
        def insert_once_text_backward(text_backward_micro_batch_id):
            light_mb_id = light_pipeline_id * light_num_microbatches + text_backward_micro_batch_id
            self._inst.append(PipelineInstruction(Ops.recv_backward, 1, light_mb_id, "up"))
            self._inst.append(PipelineInstruction(Ops.backward, 1, light_mb_id, "up"))
            self.prev_fw_bw_inst = PipelineInstruction(Ops.backward, 1, light_mb_id, "up")
            self._inst.append(PipelineInstruction(Ops.send_backward, 1, light_mb_id, "up"))
            self._inst.append(PipelineInstruction(Ops.bw_end, 1, light_mb_id, "up"))
        while text_backward_micro_batch_id < light_num_microbatches:
            if not done_intensive_text_backward:
                # send last vision backward for batched text backward
                for i in range(intensive_text_backward_num[rank]):
                    insert_once_text_backward(text_backward_micro_batch_id)
                    if i == intensive_text_backward_num[rank] - 1:
                        last_1f1b_send_backward = last_1f1b_backward_output.pop()
                        self._inst.append(last_1f1b_send_backward)
                        self._inst.append(PipelineInstruction(Ops.bw_end, last_1f1b_send_backward.model_chunk_id, last_1f1b_send_backward.micro_batch_id, last_1f1b_send_backward.up_or_down))
                    text_backward_micro_batch_id += 1
                done_intensive_text_backward = True
            else:
                # no batched text backward
                insert_once_text_backward(text_backward_micro_batch_id)
                text_backward_micro_batch_id += 1
        # Some ranks (e.g. 0,1,2) finish cool down very rapid, will leave text backward values to be sent.
        while len(send_up_backward_pend) > 0:
            send_up_inst = send_up_backward_pend.pop(0)
            self._inst.append(send_up_inst)
            self._inst.append(PipelineInstruction(Ops.bw_end, 1, send_up_inst.micro_batch_id, "up"))
        return self._inst

if __name__ == "__main__":
    sches = Schedules4()
    sches.generate(rank=4, pipeline_parallel_world_size=8, num_microbatches=8, real_modal_ratio=2, fuse_light_factor=1, intensive_backward_k=2, duplicate_factor=1)
    sches.format_schedule()
