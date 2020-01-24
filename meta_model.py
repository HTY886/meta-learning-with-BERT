from wrapper import NoWrapper, FOMAMLWrapper, ReptileWrapper

def get_meta_model(args, model, optimizer=None, scheduler=None):
    """Wrap model into meta-model"""

    if args.meta_model.lower() == 'no':
        return NoWrapper(
            args,
            model,
            args.inner_opt,
            args.inner_kwargs
        )

    if args.meta_model.lower() == 'fomaml':
        return FOMAMLWrapper(
            args,
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            optimizer=optimizer,
            scheduler=scheduler
        )

    if args.meta_model.lower() == 'reptile':
        return ReptileWrapper(
            args,
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            optimizer=optimizer,
            scheduler=scheduler
        )

    raise NotImplementedError('Meta-learner {} unknown.'.format(args.meta_model.lower()))

