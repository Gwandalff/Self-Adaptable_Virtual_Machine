package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.interpreter.miniJava.Block;
import miniJava.interpreter.miniJava.BooleanValue;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.WhileStatement;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;

public class WhileStatementImpl extends StatementImpl implements WhileStatement {
	protected Expression condition;

	protected Block block;

	protected WhileStatementImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.WHILE_STATEMENT;
	}

	public Expression getCondition() {
		return condition;
	}

	public NotificationChain basicSetCondition(Expression newCondition, NotificationChain msgs) {
		Expression oldCondition = condition;
		condition = newCondition;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.WHILE_STATEMENT__CONDITION, oldCondition, newCondition);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setCondition(Expression newCondition) {
		if (newCondition != condition) {
			NotificationChain msgs = null;
			if (condition != null)
				msgs = ((InternalEObject) condition).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.WHILE_STATEMENT__CONDITION, null, msgs);
			if (newCondition != null)
				msgs = ((InternalEObject) newCondition).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.WHILE_STATEMENT__CONDITION, null, msgs);
			msgs = basicSetCondition(newCondition, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.WHILE_STATEMENT__CONDITION, newCondition, newCondition));
	}

	public Block getBlock() {
		return block;
	}

	public NotificationChain basicSetBlock(Block newBlock, NotificationChain msgs) {
		Block oldBlock = block;
		block = newBlock;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.WHILE_STATEMENT__BLOCK, oldBlock, newBlock);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setBlock(Block newBlock) {
		if (newBlock != block) {
			NotificationChain msgs = null;
			if (block != null)
				msgs = ((InternalEObject) block).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.WHILE_STATEMENT__BLOCK, null, msgs);
			if (newBlock != null)
				msgs = ((InternalEObject) newBlock).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.WHILE_STATEMENT__BLOCK, null, msgs);
			msgs = basicSetBlock(newBlock, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.WHILE_STATEMENT__BLOCK, newBlock, newBlock));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.WHILE_STATEMENT__CONDITION :
				return basicSetCondition(null, msgs);
			case MiniJavaPackage.WHILE_STATEMENT__BLOCK :
				return basicSetBlock(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.WHILE_STATEMENT__CONDITION :
				return getCondition();
			case MiniJavaPackage.WHILE_STATEMENT__BLOCK :
				return getBlock();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.WHILE_STATEMENT__CONDITION :
				setCondition((Expression) newValue);
				return;
			case MiniJavaPackage.WHILE_STATEMENT__BLOCK :
				setBlock((Block) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.WHILE_STATEMENT__CONDITION :
				setCondition((Expression) null);
				return;
			case MiniJavaPackage.WHILE_STATEMENT__BLOCK :
				setBlock((Block) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.WHILE_STATEMENT__CONDITION :
				return condition != null;
			case MiniJavaPackage.WHILE_STATEMENT__BLOCK :
				return block != null;
		}
		return super.eIsSet(featureID);
	}

	public void evaluateStatement(State state) {
		BooleanValue continueWhile = ((BooleanValue) (((BooleanValue) (((Expression) (this.getCondition())).evaluateExpression((State) (state))))));
		while (continueWhile.isValue()) {
			((Block) (this.getBlock())).evaluateStatement((State) (state));
			BooleanValue continueWhile2 = ((BooleanValue) (((BooleanValue) (((Expression) (this.getCondition())).evaluateExpression((State) (state))))));
			continueWhile = continueWhile2;
		}
	}
}
