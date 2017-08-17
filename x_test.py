import ld_gan


BATCH_SIZE  = 128
LATENT_SIZE = 100
LEARNING_RATE = 0.0002
RAND_SEED = 42


X, Y = ld_gan.data_proc.data_loader.load_data(1, verbose=1, resize = 64)


cudnn.benchmark = True
torch.cuda.manual_seed_all(RAND_SEED)


gen = ld_gan.models.gen.gen_64()
dis = ld_gan.models.dis.dis_64()

train_ops = [ld_gan.train_ops.GanDis(gen, dis, LEARNING_RATE),
             ld_gan.train_ops.GanGen(gen, dis, LEARNING_RATE)]


sampler = ld_gan.sample.generate_rand_noise(X, Y, BATCH_SIZE, LATENT_SIZE)


trainer = ld_gan.trainer.Trainer(gen, 
                                 dis, 
                                 None, 
                                 train_ops,
                                 sampler,
                                 len(X))

trainer.train()