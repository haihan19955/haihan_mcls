model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b3'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        data_prefix=
        '/home/haihan/projects/mmlab/mmclassification/data/imagenet/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                size=300,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CustomDataset',
        data_prefix=
        '/home/haihan/projects/mmlab/mmclassification/data/imagenet/val',
        ann_file=
        '/home/haihan/projects/mmlab/mmclassification/data/imagenet/meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='CenterCrop',
                crop_size=300,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='CustomDataset',
        data_prefix=
        '/home/haihan/projects/mmlab/mmclassification/data/imagenet/val',
        ann_file=
        '/home/haihan/projects/mmlab/mmclassification/data/imagenet/meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='CenterCrop',
                crop_size=300,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[50])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/haihan/projects/mmlab/checkpoints/efficientnet-b3_3rdparty_8xb32_in1k_20220119-4b4d7487.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = '.\work_dirs\efficientnet_b3_flower_batch-16_lr-step-15_test_haihanflower'
gpu_ids = range(0, 1)
