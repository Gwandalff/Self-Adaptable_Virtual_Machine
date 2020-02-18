package visitor;

public interface IDynamicModule {
	boolean update(AcceptInterface self);
	int[] targetedNodes();
}
